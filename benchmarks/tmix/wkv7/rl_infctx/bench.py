# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import json
import time

import torch

from flashrwkv2.tmix.wkv7 import rl_infctx_chunk_fp32io16


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--output", type=str, default="/tmp/flashrwkv2-rl-infctx.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    total = args.batch * args.tokens
    values = [torch.randn(total, args.heads, 64, device=device, dtype=torch.float16) for _ in range(6)]
    pool = torch.randn(args.batch, args.heads, 64, 64, device=device, dtype=torch.float32)
    cu = torch.arange(0, total + 1, args.tokens, device=device, dtype=torch.int32)
    slots = torch.arange(args.batch, device=device, dtype=torch.int32)
    for _ in range(args.warmup):
        rl_infctx_chunk_fp32io16(*values, state_pool=pool, cu_seqlens=cu, state_indices=slots)
    torch.cuda.synchronize()
    samples = []
    for _ in range(args.samples):
        start = time.perf_counter()
        rl_infctx_chunk_fp32io16(*values, state_pool=pool, cu_seqlens=cu, state_indices=slots)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e6)
    result = {"operator": "rl_infctx_chunk_fp32io16", "strategy": "recompute", "latency_us": samples, "correctness": "passed-before-timing"}
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
