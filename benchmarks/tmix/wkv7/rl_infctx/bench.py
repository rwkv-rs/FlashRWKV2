# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import json

import torch
from _timing import measure_cuda

from flashrwkv2.tmix.wkv7 import rl_infctx_tmix_wkv7_chunk_fp32io16


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--heads", type=int, default=2)
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
    def run() -> tuple[torch.Tensor, torch.Tensor]:
        return rl_infctx_tmix_wkv7_chunk_fp32io16(
            *values, state_pool=pool, cu_seqlens=cu, state_indices=slots
        )

    timing = measure_cuda(run, before_batch=lambda: pool.copy_(initial_pool))
    result = {
        "operator": "rl_infctx_tmix_wkv7_chunk_fp32io16",
        "profile": (
            f"batch={args.batch}/tokens={args.tokens}/heads={args.heads}/d=64"
        ),
        "strategy": "recompute",
        "steady_state": "state pool is reset before each timing batch",
        **timing,
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
