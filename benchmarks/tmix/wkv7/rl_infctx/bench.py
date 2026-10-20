# SPDX-License-Identifier: MIT
"""Packed BF16 RL/Infctx with differentiable FP32 state."""

import argparse
import json

import torch
from _timing import measure_cuda

from flashrwkv2 import rl_infctx_tmix_wkv7_recurrent_bf16


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--head-size", type=int, choices=(64, 128, 256), default=64)
    parser.add_argument("--backward", action="store_true")
    args = parser.parse_args()
    total, dim = args.batch * args.tokens, args.head_size
    values = [(torch.randn(total, args.heads, dim, device="cuda") * 0.05).bfloat16().requires_grad_(args.backward) for _ in range(6)]
    pool = (torch.randn(args.batch, args.heads, dim, dim, device="cuda") * 0.01).requires_grad_(args.backward)
    cu = torch.arange(args.batch + 1, device="cuda", dtype=torch.int32) * args.tokens
    dy, ds = torch.randn_like(values[0]), torch.randn_like(pool)

    def run():
        output, final = rl_infctx_tmix_wkv7_recurrent_bf16(*values, state_pool=pool, cu_seqlens=cu)
        if args.backward:
            torch.autograd.grad((output, final), (*values, pool), (dy, ds))
        return output, final

    print(json.dumps({
        "operator": "rl_infctx_tmix_wkv7_recurrent_bf16",
        "profile": f"batch={args.batch}/tokens={args.tokens}/heads={args.heads}/d={dim}/backward={args.backward}",
        "steady_state": "functional state inputs are reused; autograd graph is released per launch",
        **measure_cuda(run),
    }))


if __name__ == "__main__":
    main()
