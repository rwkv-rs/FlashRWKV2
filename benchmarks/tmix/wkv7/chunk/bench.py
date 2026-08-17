# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import json
import time

import torch

from flashrwkv2.tmix.wkv7 import infer_tmix_wkv7_chunk_bf16_forward_varlen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seqlen", type=int, default=32)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()
    device = torch.device("cuda")
    total = args.batch * args.seqlen
    inputs = [
        torch.randn(total, args.heads, 64, device=device, dtype=torch.bfloat16)
        for _ in range(6)
    ]
    state_pool = torch.zeros(
        args.batch, args.heads, 64, 64, device=device, dtype=torch.bfloat16
    )
    cu_seqlens = torch.arange(
        0, total + 1, args.seqlen, device=device, dtype=torch.int32
    )
    state_indices = torch.arange(args.batch, device=device, dtype=torch.int32)
    for _ in range(10):
        infer_tmix_wkv7_chunk_bf16_forward_varlen(
            *inputs,
            state_pool=state_pool,
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
            chunk_size=16,
            max_seqlen=args.seqlen,
        )
    torch.cuda.synchronize()
    samples = []
    for _ in range(args.iters):
        start = time.perf_counter()
        infer_tmix_wkv7_chunk_bf16_forward_varlen(
            *inputs,
            state_pool=state_pool,
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
            chunk_size=16,
            max_seqlen=args.seqlen,
        )
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e6)
    samples.sort()
    print(json.dumps({
        "operator": "infer_tmix_wkv7_chunk_bf16_forward_varlen",
        "batch": args.batch,
        "seqlen": args.seqlen,
        "raw_latency_us": samples,
        "p50_us": samples[len(samples) // 2],
        "gpu": torch.cuda.get_device_name(),
    }))


if __name__ == "__main__":
    main()
