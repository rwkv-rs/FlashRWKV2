# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import json

import torch
from _timing import measure_cuda

from flashrwkv2.tmix.wkv7 import infer_tmix_wkv7_chunk_bf16_forward_varlen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seqlen", type=int, default=32)
    parser.add_argument("--heads", type=int, default=1)
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
    initial_state = state_pool.clone()

    def run() -> None:
        infer_tmix_wkv7_chunk_bf16_forward_varlen(
            *inputs,
            state_pool=state_pool,
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
            chunk_size=16,
            max_seqlen=args.seqlen,
        )

    timing = measure_cuda(run, before_batch=lambda: state_pool.copy_(initial_state))
    print(
        json.dumps(
            {
                "operator": "infer_tmix_wkv7_chunk_bf16_forward_varlen",
                "batch": args.batch,
                "seqlen": args.seqlen,
                "timing": timing,
                "steady_state": "state pool reset before each timing batch",
                "gpu": torch.cuda.get_device_name(),
            }
        )
    )


if __name__ == "__main__":
    main()
