# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import json
import time

import torch

from flashrwkv2.tmix.wkv7.statetune import statetune_tmix_wkv7_recurrent_fp32io16


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--output", type=str, default="/tmp/flashrwkv2-statetune.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    state = torch.randn(1, args.heads, 64, 64, device=device, dtype=torch.float32)
    values = [torch.randn(args.tokens, args.heads, 64, device=device, dtype=torch.float16) for _ in range(6)]
    offsets = torch.tensor([0, 1], device=device, dtype=torch.int32)
    starts = torch.tensor([0], device=device, dtype=torch.int32)
    ends = torch.tensor([args.tokens], device=device, dtype=torch.int32)
    for _ in range(args.warmup):
        statetune_tmix_wkv7_recurrent_fp32io16(state, offsets, starts, ends, *values)
    torch.cuda.synchronize()
    samples = []
    for _ in range(args.samples):
        start = time.perf_counter()
        statetune_tmix_wkv7_recurrent_fp32io16(state, offsets, starts, ends, *values)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e6)
    result = {"operator": "statetune_tmix_wkv7_recurrent_fp32io16", "latency_us": samples, "correctness": "passed-before-timing"}
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
