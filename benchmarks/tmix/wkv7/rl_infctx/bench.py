# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import json

import torch
from _timing import measure_cuda

from flashrwkv2.tmix.wkv7 import (
    _extension,
    rl_infctx_tmix_wkv7_chunk_fp32io16,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument(
        "--stage", choices=("forward", "backward_replay"), default="forward"
    )
    parser.add_argument(
        "--strategy", choices=("materialized", "recompute"), default="recompute"
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    total = args.batch * args.tokens
    values = [torch.randn(total, args.heads, 64, device=device, dtype=torch.float16) for _ in range(6)]
    pool = torch.randn(args.batch, args.heads, 64, 64, device=device, dtype=torch.float32)
    initial_pool = pool.clone()
    cu = torch.arange(0, total + 1, args.tokens, device=device, dtype=torch.int32)
    slots = torch.arange(args.batch, device=device, dtype=torch.int32)
    if args.stage == "forward":

        def run() -> tuple[torch.Tensor, torch.Tensor]:
            return rl_infctx_tmix_wkv7_chunk_fp32io16(
                *values,
                state_pool=pool,
                cu_seqlens=cu,
                state_indices=slots,
                strategy=args.strategy,
            )

        before_batch = lambda: pool.copy_(initial_pool)
    else:
        chunk_token_starts = torch.arange(
            0, total, args.tokens, device=device, dtype=torch.int32
        )
        chunk_token_ends = chunk_token_starts + args.tokens
        boundary = pool.clone()
        output = torch.empty_like(values[3])
        state_dot_a = torch.empty_like(values[0], dtype=torch.float32)

        def run() -> None:
            _extension().rl_infctx_tmix_wkv7_chunk_fp32io16_backward_replay(
                chunk_token_starts,
                chunk_token_ends,
                boundary,
                *values,
                output,
                state_dot_a,
            )

        before_batch = None

    timing = measure_cuda(run, before_batch=before_batch)
    result = {
        "operator": (
            "rl_infctx_tmix_wkv7_chunk_fp32io16"
            if args.stage == "forward"
            else "rl_infctx_tmix_wkv7_chunk_fp32io16_backward_replay"
        ),
        "profile": (
            f"stage={args.stage}/strategy={args.strategy}/batch={args.batch}/"
            f"tokens={args.tokens}/heads={args.heads}/d=64"
        ),
        "strategy": args.strategy if args.stage == "forward" else None,
        "steady_state": (
            "state pool is reset before each timing batch"
            if args.stage == "forward"
            else "preallocated replay outputs are overwritten on every launch"
        ),
        **timing,
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
