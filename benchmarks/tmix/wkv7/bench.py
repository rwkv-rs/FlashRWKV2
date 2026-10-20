# SPDX-License-Identifier: MIT
"""Recurrent inference comparison with FP32 state and a fixed logical workload.

Select the package revision through PYTHONPATH. The CUDA baseline supports
BF16 control runs and FP16 production runs through its owning FP32IO16 state.
"""

import argparse
import json
from functools import partial
from pathlib import Path

import torch
from _timing import measure_cuda

import flashrwkv2
from flashrwkv2 import (
    infer_tmix_wkv7_recurrent_fp32io16_forward_varlen,
    prepare_tmix_wkv7_recurrent_fp32io16_state,
    prepare_tmix_wkv7_recurrent_metadata,
)

SHAPES = {
    "h12d64": (12, 64),
    "h16d64": (16, 64),
    "h32d64": (32, 64),
    "h40d64": (40, 64),
    "h64d64": (64, 64),
    "h80d64": (80, 64),
    "h16d128": (16, 128),
    "h8d256": (8, 256),
}
CASES = (
    "1x1",
    "2x1",
    "4x1",
    "16x1",
    "64x1",
    "256x1",
    "1x16",
    "4x16",
    "64x16",
    "1x128",
    "4x128",
    "1x1024",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shapes", nargs="+", choices=tuple(SHAPES), default=list(SHAPES)
    )
    parser.add_argument("--cases", nargs="+", default=CASES)
    parser.add_argument("--decay-bias", action="store_true")
    parser.add_argument("--tune", action="store_true")
    parser.add_argument(
        "--implementation",
        choices=("tilelang", "cuda-baseline"),
        default="tilelang",
        help="select TileLang or the saved CUDA baseline package through PYTHONPATH",
    )
    parser.add_argument(
        "--token-dtype",
        choices=("bfloat16", "float16"),
        help="defaults to BF16 for TileLang and FP16 for the CUDA production baseline",
    )
    parser.add_argument("--timing-mode", choices=("eager", "graph"), default="eager")
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/wkv7-recurrent-bf16.json")
    )
    args = parser.parse_args()
    package_path = Path(flashrwkv2.__file__).resolve()
    is_tilelang = (package_path.parent / "_internal" / "linear.py").is_file()
    if is_tilelang != (args.implementation == "tilelang"):
        parser.error(
            "--implementation does not match the package selected by PYTHONPATH"
        )
    if args.token_dtype is None:
        args.token_dtype = "bfloat16" if is_tilelang else "float16"
    if args.implementation == "tilelang" and args.token_dtype != "bfloat16":
        parser.error("TileLang token IO is bfloat16")
    if args.implementation == "cuda-baseline" and args.tune:
        parser.error("--tune applies only to TileLang")
    dtype = getattr(torch, args.token_dtype)
    results = []
    for shape in args.shapes:
        heads, dim = SHAPES[shape]
        for case in args.cases:
            batch, length = (int(value) for value in case.split("x"))
            if batch <= 0 or length <= 0:
                parser.error("--cases must contain positive BxT dimensions")
            # Reset per case so selecting a subset or tuning does not change
            # the logical values used by a different implementation.
            torch.manual_seed(args.seed)
            values = tuple(
                (torch.randn(batch * length, heads, dim, device="cuda") * scale)
                .bfloat16()
                .half()
                .to(dtype)
                for scale in (0.1, 2.0, 0.05, 0.05, 0.05, 0.05)
            )
            initial = torch.randn(batch, heads, dim, dim, device="cuda") * 0.01
            state = prepare_tmix_wkv7_recurrent_fp32io16_state(
                batch,
                heads * dim,
                sequence_capacity=batch,
                head_size=dim,
                device="cuda",
            )
            pool = state._components()[0]

            def reset(state=state, pool=pool, initial=initial):
                state.zero_()
                pool.copy_(initial)

            reset()
            cu = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * length
            slots = torch.arange(batch, device="cuda", dtype=torch.int32)
            ticket = prepare_tmix_wkv7_recurrent_metadata(
                cu,
                slots,
                total_tokens=batch * length,
                state_pool_size=batch,
                max_seqlen=length,
            )
            bias = (
                torch.randn(heads, dim, device="cuda").bfloat16().half().to(dtype)
                if args.decay_bias
                else None
            )
            tuning = None
            if args.tune:
                from flashrwkv2.tmix.wkv7.tilelang import tune_tmix_wkv7_recurrent

                tuning = tune_tmix_wkv7_recurrent(*values, initial, cu, decay_bias=bias)

            run = partial(
                infer_tmix_wkv7_recurrent_fp32io16_forward_varlen,
                *values,
                state=state,
                cu_seqlens=cu,
                state_indices=slots,
                decay_bias=bias,
                max_seqlen=length,
                validated_metadata=ticket,
            )

            expected = run().clone()
            state.materialize_slots_(slots)
            expected_state = pool.clone()
            errors = {"output_relative_l2": 0.0, "state_relative_l2": 0.0}
            # DeltaLog and ordinary CUDA providers can differ within the
            # baseline's 4e-3 relative-L2 contract. TileLang must be bitwise equal.
            for row in range(batch):
                single = prepare_tmix_wkv7_recurrent_fp32io16_state(
                    1,
                    heads * dim,
                    sequence_capacity=1,
                    head_size=dim,
                    device="cuda",
                )
                single.zero_()
                single_pool = single._components()[0]
                single_pool.copy_(initial[row : row + 1])
                offsets = torch.tensor([0, length], device="cuda", dtype=torch.int32)
                slot = torch.zeros(1, device="cuda", dtype=torch.int32)
                single_ticket = prepare_tmix_wkv7_recurrent_metadata(
                    offsets,
                    slot,
                    total_tokens=length,
                    state_pool_size=1,
                    max_seqlen=length,
                )
                actual = infer_tmix_wkv7_recurrent_fp32io16_forward_varlen(
                    *(
                        value[row * length : (row + 1) * length].contiguous()
                        for value in values
                    ),
                    state=single,
                    cu_seqlens=offsets,
                    state_indices=slot,
                    decay_bias=bias,
                    max_seqlen=length,
                    validated_metadata=single_ticket,
                )
                single.materialize_slots_(slot)
                for name, result, reference in (
                    (
                        "output_relative_l2",
                        actual,
                        expected[row * length : (row + 1) * length],
                    ),
                    ("state_relative_l2", single_pool[0], expected_state[row]),
                ):
                    error = float(
                        (result.float() - reference.float()).norm()
                        / reference.float().norm().clamp_min(1e-12)
                    )
                    errors[name] = max(errors[name], error)
                    passed = (
                        torch.equal(result, reference)
                        if args.implementation == "tilelang"
                        else error <= 4e-3
                    )
                    if not passed:
                        raise AssertionError(
                            f"batch invariance failed for {shape}/{case}: {name}={error}"
                        )
            timing = measure_cuda(
                run,
                before_batch=reset,
                timing_mode=args.timing_mode,
            )
            state.materialize_slots_(slots)
            record = {
                "operator_shape": shape,
                "case": case,
                "implementation": args.implementation,
                "seed": args.seed,
                "token_dtype": str(values[0].dtype).removeprefix("torch."),
                "decay_logits_dtype": str(values[1].dtype).removeprefix("torch."),
                "decay_bias_dtype": None
                if bias is None
                else str(bias.dtype).removeprefix("torch."),
                "output_dtype": str(expected.dtype).removeprefix("torch."),
                "state_dtype": str(pool.dtype).removeprefix("torch."),
                "state_memory_layout": state.memory_layout,
                "metadata": {
                    "total_tokens": batch * length,
                    "sequences": batch,
                    "state_pool_size": batch,
                    "max_seqlen": length,
                },
                "correctness": "batch-bitwise-passed"
                if args.implementation == "tilelang"
                else "batch-relative-l2-passed",
                "correctness_errors": errors,
                "timing": timing,
            }
            if tuning is not None:
                record["tuning"] = tuning
            results.append(record)
    result = {
        "benchmark": "tilelang_wkv7_recurrent"
        if args.implementation == "tilelang"
        else "cuda_baseline_wkv7_recurrent",
        "implementation": args.implementation,
        "package_path": str(package_path),
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": torch.cuda.get_device_capability(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "logical_inputs": "FP32 normal samples rounded through BF16 and FP16; identical values in both token formats",
        "steady_state": "owning state reset before each timing batch; metadata preparation and state materialization are untimed",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
