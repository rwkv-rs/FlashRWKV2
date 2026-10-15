# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import json

import torch
from _timing import measure_cuda

from flashrwkv2.head.linear import (
    infer_head_linear_all_forward_varlen,
    pretrain_head_linear_bf16,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--channels", type=int, default=4096)
    parser.add_argument("--vocab", type=int, default=65536)
    parser.add_argument("--adapter-rank", type=int, default=0)
    args = parser.parse_args()
    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.adapter_rank else torch.float16
    x = torch.randn(
        args.tokens, args.channels, device=device, dtype=dtype
    ).requires_grad_(bool(args.adapter_rank))
    weight = torch.randn(args.vocab, args.channels, device=device, dtype=dtype)
    if args.adapter_rank:
        a = (
            torch.randn(args.adapter_rank, args.channels, device=device) * 0.01
        ).requires_grad_()
        b = (
            torch.randn(args.vocab, args.adapter_rank, device=device) * 0.01
        ).requires_grad_()
        grad = torch.randn(args.tokens, args.vocab, device=device, dtype=dtype)

        def run():
            out = pretrain_head_linear_bf16(
                x, weight, lora_a=a, lora_b=b, use_nora=True
            )
            torch.autograd.grad(out, (x, a, b), grad)
            return out
    else:

        def run():
            return infer_head_linear_all_forward_varlen(x, weight)

    timing = measure_cuda(run)
    print(
        json.dumps(
            {
                "operator": "pretrain_head_linear_bf16"
                if args.adapter_rank
                else "infer_head_linear_all_forward_varlen",
                "profile": (
                    f"tokens={args.tokens}/channels={args.channels}/vocab={args.vocab}/adapter_rank={args.adapter_rank}"
                ),
                "steady_state": "stateless inputs are reused",
                **timing,
            }
        )
    )


if __name__ == "__main__":
    main()
