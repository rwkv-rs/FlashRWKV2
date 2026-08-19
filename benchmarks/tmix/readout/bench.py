# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import json

import torch
from _timing import measure_cuda

from flashrwkv2.tmix.readout import (
    infer_tmix_readout_forward_varlen,
    pretrain_tmix_readout_bf16,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--channels", type=int, default=4096)
    parser.add_argument(
        "--operator", choices=("infer", "pretrain"), default="infer"
    )
    parser.add_argument("--backward", action="store_true")
    args = parser.parse_args()
    if args.operator == "infer" and args.backward:
        raise ValueError("the inference readout does not support backward")

    if args.operator == "infer":
        options = {"device": torch.device("cuda"), "dtype": torch.float16}
        packed = [
            torch.randn(args.tokens, args.channels, **options) for _ in range(5)
        ]
        affine = [torch.randn(args.channels, **options) for _ in range(3)]
        output_weight = torch.randn(args.channels, args.channels, **options) * 0.01

        def run() -> torch.Tensor:
            return infer_tmix_readout_forward_varlen(
                packed[0],
                packed[1],
                packed[2],
                packed[3],
                affine[0],
                affine[1],
                affine[2],
                packed[4],
                output_weight,
                head_size=64,
                batch_size=1,
                max_seqlen=args.tokens,
            )

    else:
        shape = (1, args.tokens, args.channels)
        tokens = tuple(
            torch.randn(
                shape,
                device="cuda",
                dtype=torch.bfloat16,
                requires_grad=args.backward,
            )
            for _ in range(4)
        )
        heads = args.channels // 64
        residual_scale = torch.randn(
            heads,
            64,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=args.backward,
        )
        affine = tuple(
            torch.randn(
                args.channels,
                device="cuda",
                dtype=torch.bfloat16,
                requires_grad=args.backward,
            )
            for _ in range(2)
        )
        gate = torch.randn(
            shape,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=args.backward,
        )
        inputs = (*tokens, residual_scale, *affine, gate)
        grad_output = torch.randn_like(tokens[0])

        def run() -> torch.Tensor:
            output = pretrain_tmix_readout_bf16(*inputs)
            if args.backward:
                torch.autograd.grad(output, inputs, grad_output)
            return output

    timing = measure_cuda(run)
    print(
        json.dumps(
            {
                "operator": args.operator,
                "forward_backward": args.backward,
                "profile": (
                    f"operator={args.operator}/backward={args.backward}/"
                    f"tokens={args.tokens}/channels={args.channels}"
                ),
                "steady_state": "stateless inputs are reused",
                **timing,
            }
        )
    )


if __name__ == "__main__":
    main()
