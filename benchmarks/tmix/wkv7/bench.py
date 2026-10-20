# SPDX-License-Identifier: MIT
"""BF16 recurrent inference matrix, including exact batch-invariance checks."""

import argparse
import json
from functools import partial
from pathlib import Path

import torch
from _timing import measure_cuda

from flashrwkv2 import (
    infer_tmix_wkv7_recurrent_fp32io16_forward_varlen,
    prepare_tmix_wkv7_recurrent_fp32io16_state_from_tensor,
    prepare_tmix_wkv7_recurrent_metadata,
)
from flashrwkv2.tmix.wkv7.tilelang import tune_tmix_wkv7_recurrent

SHAPES = {
    "h12d64": (12, 64),
    "h32d64": (32, 64),
    "h64d64": (64, 64),
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
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/wkv7-recurrent-bf16.json")
    )
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    results = []
    for shape in args.shapes:
        heads, dim = SHAPES[shape]
        for case in args.cases:
            batch, length = (int(value) for value in case.split("x"))
            values = tuple(
                (
                    torch.randn(batch * length, heads, dim, device="cuda") * scale
                ).bfloat16()
                for scale in (0.1, 2.0, 0.05, 0.05, 0.05, 0.05)
            )
            initial = torch.randn(batch, heads, dim, dim, device="cuda") * 0.01
            pool = initial.clone()
            state = prepare_tmix_wkv7_recurrent_fp32io16_state_from_tensor(pool)
            cu = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * length
            slots = torch.arange(batch, device="cuda", dtype=torch.int32)
            ticket = prepare_tmix_wkv7_recurrent_metadata(
                cu, slots, total_tokens=batch * length, state_pool_size=batch
            )
            bias = (
                torch.randn(heads, dim, device="cuda", dtype=torch.bfloat16)
                if args.decay_bias
                else None
            )
            tuning = (
                tune_tmix_wkv7_recurrent(*values, initial, cu, decay_bias=bias)
                if args.tune
                else None
            )

            run = partial(
                infer_tmix_wkv7_recurrent_fp32io16_forward_varlen,
                *values,
                state=state,
                cu_seqlens=cu,
                state_indices=slots,
                decay_bias=bias,
                validated_metadata=ticket,
            )

            expected = run().clone()
            expected_state = pool.clone()
            # Each row is also checked in a one-sequence launch. This guards
            # batch-dependent tuning before any timing is reported.
            for row in range(batch):
                single_pool = initial[row : row + 1].clone()
                single = prepare_tmix_wkv7_recurrent_fp32io16_state_from_tensor(
                    single_pool
                )
                offsets = torch.tensor([0, length], device="cuda", dtype=torch.int32)
                slot = torch.zeros(1, device="cuda", dtype=torch.int32)
                actual = infer_tmix_wkv7_recurrent_fp32io16_forward_varlen(
                    *(
                        value[row * length : (row + 1) * length].contiguous()
                        for value in values
                    ),
                    state=single,
                    cu_seqlens=offsets,
                    state_indices=slot,
                    decay_bias=bias,
                )
                if not torch.equal(
                    actual, expected[row * length : (row + 1) * length]
                ) or not torch.equal(single_pool[0], expected_state[row]):
                    raise AssertionError(f"batch invariance failed for {shape}/{case}")
            timing = measure_cuda(run, before_batch=partial(pool.copy_, initial))
            record = {
                "operator_shape": shape,
                "case": case,
                "token_dtype": "bfloat16",
                "state_dtype": "float32",
                "correctness": "batch-bitwise-passed",
                "timing": timing,
            }
            if tuning is not None:
                record["tuning"] = tuning
            results.append(record)
    result = {
        "benchmark": "tilelang_wkv7_recurrent",
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": torch.cuda.get_device_capability(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "steady_state": "state reset before each timing batch",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
