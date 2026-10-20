# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import json

import torch
from _timing import measure_cuda

from flashrwkv2._internal.linear import _NoRA, linear, tune_linear
from flashrwkv2.head.linear import (
    infer_head_linear_all_forward_varlen,
    pretrain_head_linear_bf16,
)


def benchmark(args):
    backward = args.backward or bool(args.adapter_rank)
    x = torch.randn(
        args.tokens, args.channels, device="cuda", dtype=torch.bfloat16
    ).requires_grad_(backward)
    weight = torch.randn(
        args.vocab, args.channels, device="cuda", dtype=torch.bfloat16
    ).requires_grad_(backward and not args.adapter_rank)
    adapters = {}
    inputs = (x, weight)
    if args.adapter_rank:
        a = (
            torch.randn(args.adapter_rank, args.channels, device="cuda") * 0.01
        ).requires_grad_()
        b = (
            torch.randn(args.vocab, args.adapter_rank, device="cuda") * 0.01
        ).requires_grad_()
        adapters = {"lora_a": a, "lora_b": b, "use_nora": True}
        inputs = (x, a, b)
    tuning = {}
    if args.tune:
        tuning["base"] = tune_linear(x, weight, torch.float32)
        if adapters:
            normalized = _NoRA.apply(a.detach())
            tuning["adapter_a"] = tune_linear(x, normalized)
            features = linear(x.detach(), normalized)
            tuning["adapter_b"] = tune_linear(
                features, b.detach().bfloat16(), torch.float32
            )
    # The common head emits FP32 logits, including the externally supplied VJP.
    grad = (
        torch.randn(args.tokens, args.vocab, device="cuda", dtype=torch.float32)
        if backward
        else None
    )

    def run():
        if backward:
            output = pretrain_head_linear_bf16(x, weight, **adapters)
            torch.autograd.grad(output, inputs, grad)
        else:
            output = infer_head_linear_all_forward_varlen(x, weight)
        return output

    return {
        "operator": "pretrain_head_linear_bf16"
        if backward
        else "infer_head_linear_all_forward_varlen",
        "profile": f"tokens={args.tokens}/channels={args.channels}/vocab={args.vocab}/adapter_rank={args.adapter_rank}",
        "tokens": args.tokens,
        "channels": args.channels,
        "vocab": args.vocab,
        "input_dtype": "bfloat16",
        "output_dtype": "float32",
        "accumulation_dtype": "float32",
        "forward_backward": backward,
        "tuning": tuning,
        "steady_state": "stateless inputs are reused",
        **measure_cuda(run),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--channels", type=int, default=4096)
    parser.add_argument("--vocab", type=int, default=65536)
    parser.add_argument(
        "--adapter-rank", type=int, default=0, help="enable NoRA forward+backward"
    )
    parser.add_argument("--backward", action="store_true")
    parser.add_argument(
        "--tune",
        action="store_true",
        help="verify and measure output tiles before timing",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="emit JSONL for six model widths and M=1/27/256",
    )
    args = parser.parse_args()
    cases = [(args.tokens, args.channels)]
    if args.sweep:
        cases = [
            (rows, channels)
            for channels in (768, 1024, 2048, 2560, 4096, 5120)
            for rows in (1, 27, 256)
        ]
    for tokens, channels in cases:
        case = argparse.Namespace(
            **{**vars(args), "tokens": tokens, "channels": channels}
        )
        print(json.dumps(benchmark(case)))


if __name__ == "__main__":
    main()
