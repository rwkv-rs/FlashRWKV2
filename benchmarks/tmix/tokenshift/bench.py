# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import json

import torch
from _timing import measure_cuda

from flashrwkv2.tmix.tokenshift import (
    infer_tmix_postnorm_tokenshift_forward_varlen,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seqlen", type=int, default=1)
    parser.add_argument("--channels", type=int, default=4096)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.batch <= 0 or args.seqlen <= 0 or args.channels <= 0:
        raise ValueError("batch, seqlen and channels must be positive")

    device = torch.device("cuda")
    rows = args.batch * args.seqlen
    x = torch.zeros(rows, args.channels, device=device, dtype=torch.float16)
    res = torch.zeros_like(x)
    weight = torch.ones(args.channels, device=device, dtype=torch.float16)
    bias = torch.zeros_like(weight)
    parameters = [torch.zeros_like(weight) for _ in range(6)]
    shift_state = torch.zeros(
        args.batch, args.channels, device=device, dtype=torch.float16
    )
    cu_seqlens = torch.arange(
        0, rows + 1, args.seqlen, device=device, dtype=torch.int32
    )
    state_indices = torch.arange(args.batch, device=device, dtype=torch.int32)

    def run() -> tuple[torch.Tensor, ...]:
        return infer_tmix_postnorm_tokenshift_forward_varlen(
            x,
            res,
            weight,
            bias,
            *parameters,
            shift_state_pool=shift_state,
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
            max_seqlen=args.seqlen,
        )

    timing = measure_cuda(run, before_batch=shift_state.zero_)
    print(
        json.dumps(
            {
                "operator": "infer_tmix_postnorm_tokenshift_forward_varlen",
                "batch": args.batch,
                "seqlen": args.seqlen,
                "channels": args.channels,
                "profile": (
                    f"batch={args.batch}/seqlen={args.seqlen}/channels={args.channels}"
                ),
                "steady_state": "shift state is reset before each timing batch",
                **timing,
            }
        )
    )


if __name__ == "__main__":
    main()
