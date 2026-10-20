# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from _timing import measure_cuda

import flashrwkv2
from flashrwkv2.tmix.readout import (
    infer_tmix_readout_forward_varlen,
    pretrain_tmix_readout_bf16,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--channels", type=int, default=4096)
    parser.add_argument("--operator", choices=("infer", "pretrain"), default="infer")
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--adapter-rank", type=int, default=0)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--timing-mode", choices=("eager", "graph"), default="eager")
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument(
        "--implementation", choices=("tilelang", "cuda-baseline"), default="tilelang"
    )
    args = parser.parse_args()
    tilelang = args.implementation == "tilelang"
    package_path = Path(flashrwkv2.__file__).resolve()
    if (package_path.parent / "_internal" / "linear.py").is_file() != tilelang:
        raise ValueError("--implementation must match the imported flashrwkv2 package")
    if args.tune and not tilelang:
        raise ValueError("--tune requires the TileLang implementation")
    torch.manual_seed(args.seed)
    if args.adapter_rank and args.operator != "pretrain":
        raise ValueError("adapter-rank requires the pretrain operator")
    if args.operator == "infer" and args.backward:
        raise ValueError("the inference readout does not support backward")

    if args.operator == "infer":
        dtype = torch.bfloat16 if tilelang else torch.float16

        def values(*shape, scale=1.0):
            return (
                (torch.randn(shape, device="cuda") * scale).bfloat16().half().to(dtype)
            )

        packed = [values(args.tokens, args.channels) for _ in range(5)]
        if not tilelang:
            packed[0] = packed[0].float()
        affine = [values(args.channels) for _ in range(3)]
        output_weight = values(args.channels, args.channels, scale=0.01)
        tune_inputs = (*packed[:4], *affine, packed[4])
        projection = {"output_weight": output_weight}

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
        tune_inputs = inputs
        grad_output = torch.randn_like(tokens[0])

        projection = {}
        grad_inputs = inputs
        if args.adapter_rank:
            output_weight = (
                torch.randn(
                    args.channels, args.channels, device="cuda", dtype=torch.bfloat16
                )
                * 0.01
            )
            a = (
                torch.randn(args.adapter_rank, args.channels, device="cuda") * 0.01
            ).requires_grad_(args.backward)
            b = (
                torch.randn(args.channels, args.adapter_rank, device="cuda") * 0.01
            ).requires_grad_(args.backward)
            projection = {
                "output_weight": output_weight,
                "output_lora_a": a,
                "output_lora_b": b,
                "output_use_nora": True,
            }
            grad_inputs = (*inputs, a, b)

        def run() -> torch.Tensor:
            output = pretrain_tmix_readout_bf16(*inputs, **projection)
            if args.backward:
                torch.autograd.grad(output, grad_inputs, grad_output)
            return output

    tuning = {}
    if args.tune:
        from flashrwkv2._internal.linear import _NoRA, linear, tune_linear
        from flashrwkv2.tmix.readout.tilelang import readout, tune_readout

        tuning["readout"] = tune_readout(*tune_inputs)
        if projection:
            with torch.no_grad():
                hidden = readout(*tune_inputs)
                tuning["projection"] = tune_linear(hidden, output_weight)
                if args.adapter_rank:
                    normalized = _NoRA.apply(a.detach())
                    tuning["adapter_a"] = tune_linear(hidden, normalized)
                    features = linear(hidden, normalized)
                    tuning["adapter_b"] = tune_linear(
                        features, b.detach().bfloat16(), torch.float32
                    )
    sample = run()
    output_dtype = str(sample.dtype).removeprefix("torch.")
    del sample
    timing = measure_cuda(run, timing_mode=args.timing_mode)
    print(
        json.dumps(
            {
                "operator": args.operator,
                "forward_backward": args.backward,
                "implementation": args.implementation,
                "package_path": str(package_path),
                "wkv_dtype": str(tune_inputs[0].dtype).removeprefix("torch."),
                "input_dtype": str(tune_inputs[1].dtype).removeprefix("torch."),
                "output_dtype": output_dtype,
                "seed": args.seed,
                "tuning": tuning,
                "profile": (
                    f"operator={args.operator}/backward={args.backward}/"
                    f"tokens={args.tokens}/channels={args.channels}/adapter_rank={args.adapter_rank}"
                ),
                "steady_state": "stateless inputs are reused",
                **timing,
            }
        )
    )


if __name__ == "__main__":
    main()
