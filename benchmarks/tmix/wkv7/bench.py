# SPDX-License-Identifier: MIT
"""Recurrent inference comparison with FP32 state and a fixed logical workload.

Select the package revision through PYTHONPATH. The CUDA baseline supports
BF16 control runs and FP16 production runs through its owning FP32IO16 state.
"""

import argparse
import json
from functools import partial
from itertools import accumulate
from pathlib import Path

import torch
from _timing import measure_cuda

import flashrwkv2
from flashrwkv2 import (
    infer_tmix_wkv7_recurrent_fp32io16_forward_varlen,
    prepare_tmix_wkv7_recurrent_fp32io16_state_from_tensor,
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
    parser.add_argument("--cases", nargs="+", default=CASES,
                        help="packed BxT terms in scheduler order, e.g. 255x1+1x256")
    parser.add_argument("--metadata", choices=("static", "live"), default="static")
    parser.add_argument("--state-pool-size", type=int, help="physical slots (defaults to the active batch)")
    parser.add_argument("--slot-order", choices=("sequential", "random"), default="sequential")
    parser.add_argument("--decay-bias", action="store_true")
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--state-dtype", choices=("float32", "float16"), default="float32")
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
    if args.timing_mode == "graph":
        torch.cuda.set_stream(torch.cuda.Stream())
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
    if not is_tilelang and args.state_dtype != "float32":
        parser.error("the saved CUDA comparison uses FP32 state")
    prepare_state = prepare_tmix_wkv7_recurrent_fp32io16_state_from_tensor
    infer = infer_tmix_wkv7_recurrent_fp32io16_forward_varlen
    if args.state_dtype == "float16":
        from flashrwkv2 import (
            infer_tmix_wkv7_recurrent_fp16_forward_varlen,
            prepare_tmix_wkv7_recurrent_fp16_state_from_tensor,
        )
        prepare_state = prepare_tmix_wkv7_recurrent_fp16_state_from_tensor
        infer = infer_tmix_wkv7_recurrent_fp16_forward_varlen
    dtype = getattr(torch, args.token_dtype)
    results = []
    for shape in args.shapes:
        heads, dim = SHAPES[shape]
        for case in args.cases:
            lengths = []
            for term in case.split("+"):
                count, size = (int(value) for value in term.split("x"))
                if count <= 0 or size <= 0:
                    parser.error("--cases must contain positive BxT dimensions")
                lengths.extend([size] * count)
            offsets_cpu = [0, *accumulate(lengths)]
            batch, length, tokens = len(lengths), max(lengths), offsets_cpu[-1]
            pool_size = batch if args.state_pool_size is None else args.state_pool_size
            if pool_size < batch:
                parser.error("--state-pool-size must cover every active sequence")
            # Reset per case so selecting a subset or tuning does not change
            # the logical values used by a different implementation.
            torch.manual_seed(args.seed)
            values = tuple(
                (torch.randn(tokens, heads, dim, device="cuda") * scale)
                .bfloat16()
                .half()
                .to(dtype)
                for scale in (0.1, 2.0, 0.05, 0.05, 0.05, 0.05)
            )
            initial = torch.randn(pool_size, heads, dim, dim, device="cuda") * 0.01
            if args.state_dtype == "float16":
                initial = initial.half()
            state = prepare_state(initial.clone())
            pool = state._components()[0]

            def reset(state=state, pool=pool, initial=initial):
                state.zero_()
                pool.copy_(initial)

            reset()
            cu = torch.tensor(offsets_cpu, device="cuda", dtype=torch.int32)
            slots = (torch.randperm(pool_size, device="cuda", dtype=torch.int32)[:batch]
                     if args.slot_order == "random" else torch.arange(batch, device="cuda", dtype=torch.int32))
            slot_ids = slots.tolist()
            if args.metadata == "live":
                metadata_args = dict(
                    token_capacity=tokens, sequence_capacity=batch, max_seqlen_capacity=length,
                    num_active_tokens=torch.tensor([tokens], device="cuda", dtype=torch.int32),
                    num_active_sequences=torch.tensor([batch], device="cuda", dtype=torch.int32),
                )
            else:
                metadata_args = dict(total_tokens=tokens, max_seqlen=length)
            ticket = prepare_tmix_wkv7_recurrent_metadata(cu, slots, state_pool_size=pool_size, **metadata_args)
            bias = (
                torch.randn(heads, dim, device="cuda").bfloat16().half().to(dtype)
                if args.decay_bias
                else None
            )
            tuning = None
            if args.tune:
                from flashrwkv2.tmix.wkv7.tilelang import tune_tmix_wkv7_recurrent

                tuning = tune_tmix_wkv7_recurrent(
                    *values,
                    initial,
                    cu,
                    decay_bias=bias,
                    indexed=True,
                )

            run = partial(
                infer,
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
            if pool_size > batch:
                unused = torch.ones(pool_size, device="cuda", dtype=torch.bool)
                unused[slots.long()] = False
                assert torch.equal(pool[unused], initial[unused])
            errors = {"output_relative_l2": 0.0, "state_relative_l2": 0.0}
            # DeltaLog and ordinary CUDA providers can differ within the
            # baseline's 4e-3 relative-L2 contract. TileLang must be bitwise equal.
            for row, size in enumerate(lengths):
                slot_id = slot_ids[row]
                single_pool = initial[slot_id : slot_id + 1].clone()
                single = prepare_state(single_pool)
                offsets = torch.tensor([0, size], device="cuda", dtype=torch.int32)
                slot = torch.zeros(1, device="cuda", dtype=torch.int32)
                single_ticket = prepare_tmix_wkv7_recurrent_metadata(
                    offsets,
                    slot,
                    total_tokens=size,
                    state_pool_size=1,
                    max_seqlen=size,
                )
                actual = infer(
                    *(
                        value[offsets_cpu[row] : offsets_cpu[row + 1]].contiguous()
                        for value in values
                    ),
                    state=single,
                    cu_seqlens=offsets,
                    state_indices=slot,
                    decay_bias=bias,
                    max_seqlen=size,
                    validated_metadata=single_ticket,
                )
                single.materialize_slots_(slot)
                for name, result, reference in (
                    (
                        "output_relative_l2",
                        actual,
                        expected[offsets_cpu[row] : offsets_cpu[row + 1]],
                    ),
                    ("state_relative_l2", single_pool[0], expected_state[slot_id]),
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
                    "total_tokens": tokens,
                    "ticket": args.metadata, "slot_order": args.slot_order,
                    "sequences": batch,
                    "state_pool_size": pool_size,
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
