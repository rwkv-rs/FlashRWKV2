# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import json

import torch
from _timing import measure_cuda

from flashrwkv2.tmix.wkv_prepare import (
    infer_tmix_wkv_prepare_forward_varlen,
    pretrain_tmix_wkv_prepare_bf16,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--channels", type=int, default=4096)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--adapter-rank", type=int, default=0)
    args = parser.parse_args()
    options = {"device": torch.device("cuda"), "dtype": torch.float16}
    shifted = [torch.randn(args.tokens, args.channels, **options) for _ in range(6)]
    dense = [
        torch.randn(args.channels, args.channels, **options) * 0.01 for _ in range(3)
    ]
    rank_in = [
        torch.randn(args.rank, args.channels, **options) * 0.01 for _ in range(4)
    ]
    rank_out = [
        torch.randn(args.channels, args.rank, **options) * 0.01 for _ in range(4)
    ]
    runtime_in = [weight.t().contiguous() for weight in rank_in]
    runtime_out = [weight.t().contiguous() for weight in rank_out]
    vectors = [torch.zeros(args.channels, **options) for _ in range(4)]

    def run() -> tuple[torch.Tensor, ...]:
        return infer_tmix_wkv_prepare_forward_varlen(
            *shifted,
            *dense,
            *rank_in,
            *rank_out,
            *vectors,
            w1_runtime=runtime_in[0],
            a1_runtime=runtime_in[1],
            g1_runtime=runtime_in[2],
            v1_runtime=runtime_in[3],
            w2_runtime=runtime_out[0],
            a2_runtime=runtime_out[1],
            g2_runtime=runtime_out[2],
            v2_runtime=runtime_out[3],
            head_size=64,
            batch_size=1,
            max_seqlen=args.tokens,
        )

    if args.adapter_rank:
        shifted = [x.to(torch.bfloat16).unsqueeze(0).requires_grad_() for x in shifted]
        dense = [w.to(torch.bfloat16) for w in dense]
        w1, a1, g1 = [w.to(torch.bfloat16) for w in runtime_in[:3]]
        w2, a2, g2 = [w.to(torch.bfloat16) for w in runtime_out[:3]]
        w0, a0, kk, ka = [v.to(torch.bfloat16) for v in vectors]
        kk.fill_(1.0)
        adapters = [
            (torch.randn(*shape, device="cuda") * 0.01).requires_grad_()
            for shape in (
                (args.adapter_rank, args.channels),
                (args.channels, args.adapter_rank),
            )
            * 3
        ]

        def run():
            outputs = pretrain_tmix_wkv_prepare_bf16(
                *shifted,
                *dense,
                w0,
                w1,
                w2,
                a0,
                a1,
                a2,
                g1,
                g2,
                kk,
                ka,
                receptance_lora_a=adapters[0],
                receptance_lora_b=adapters[1],
                key_lora_a=adapters[2],
                key_lora_b=adapters[3],
                value_lora_a=adapters[4],
                value_lora_b=adapters[5],
                receptance_use_nora=True,
                key_use_nora=True,
                value_use_nora=True,
            )
            torch.autograd.grad(
                sum(x.float().sum() for x in outputs), (*shifted, *adapters)
            )
            return outputs

    timing = measure_cuda(run)
    print(
        json.dumps(
            {
                "operator": "pretrain_tmix_wkv_prepare_bf16"
                if args.adapter_rank
                else "infer_tmix_wkv_prepare_forward_varlen",
                "profile": (
                    f"tokens={args.tokens}/channels={args.channels}/rank={args.rank}/adapter_rank={args.adapter_rank}"
                ),
                "steady_state": "stateless inputs and weights are reused",
                **timing,
            }
        )
    )


if __name__ == "__main__":
    main()
