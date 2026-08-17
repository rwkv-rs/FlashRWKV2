# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import time

import torch

from flashrwkv2.tmix.vres_gate import pretrain_tmix_vres_gate_bf16


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seqlen", type=int, default=32)
    parser.add_argument("--channels", type=int, default=4096)
    parser.add_argument("--samples", type=int, default=30)
    args = parser.parse_args()
    shape = (args.batch, args.seqlen, args.channels)
    value = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    first = torch.randn_like(value)
    v0 = torch.randn(args.channels, device="cuda", dtype=torch.bfloat16)
    v12 = torch.randn_like(value)
    samples = []
    for _ in range(args.samples):
        start = time.perf_counter()
        pretrain_tmix_vres_gate_bf16(value, first, v0, v12)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e6)
    print({"operator": "pretrain_tmix_vres_gate_bf16", "raw_latency_us": samples})


if __name__ == "__main__":
    main()
