# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from _timing import measure_cuda

import flashrwkv2
from flashrwkv2.tmix.wkv_prepare import (
    infer_tmix_wkv_prepare_forward_varlen,
    pretrain_tmix_wkv_prepare_bf16,
)


def benchmark(args):
    tilelang = args.implementation == "tilelang"
    package_path = Path(flashrwkv2.__file__).resolve()
    if (package_path.parent / "_internal" / "linear.py").is_file() != tilelang:
        raise ValueError("--implementation must match the imported flashrwkv2 package")
    if args.tune and not tilelang:
        raise ValueError("--tune requires the TileLang implementation")
    operator = args.operator or ("pretrain" if args.adapter_rank else "infer")
    backward = args.backward or (args.operator is None and bool(args.adapter_rank))
    if operator == "infer" and backward:
        raise ValueError("--backward requires the pretrain operator")
    if not tilelang and operator == "infer" and args.adapter_rank:
        raise ValueError("CUDA baseline inference has no NoRA parameters")
    torch.manual_seed(args.seed)
    channels, rank = args.channels, args.rank
    full_training = backward and not args.adapter_rank
    dtype = torch.bfloat16 if tilelang or operator == "pretrain" else torch.float16

    def random(*shape, grad=False):
        return (
            (torch.randn(shape, device="cuda") * 0.03).bfloat16().half().to(dtype)
        ).requires_grad_(grad)

    shape = (
        (1, args.tokens, channels)
        if operator == "pretrain"
        else (args.tokens, channels)
    )
    shifted = [random(*shape, grad=backward) for _ in range(6)]
    dense = [random(channels, channels, grad=full_training) for _ in range(3)]
    rank_in = [random(rank, channels) for _ in range(4)]
    rank_out = [random(channels, rank) for _ in range(4)]
    runtime_in = [
        weight.t().contiguous().requires_grad_(full_training) for weight in rank_in
    ]
    runtime_out = [
        weight.t().contiguous().requires_grad_(full_training) for weight in rank_out
    ]
    vectors = [random(channels) for _ in range(4)]
    vectors[1].fill_(1.0)
    for vector in vectors:
        vector.requires_grad_(full_training)
    v0, kk, a0, ka = vectors
    w0 = random(channels, grad=full_training)
    first = random(*shape, grad=backward) if args.later_layer else None
    adapters, adapter_options = [], {}
    if args.adapter_rank:
        adapters = [
            (torch.randn(size, device="cuda") * 0.01).requires_grad_(backward)
            for size in ((args.adapter_rank, channels), (channels, args.adapter_rank))
            * 3
        ]
        for index, name in enumerate(("receptance", "key", "value")):
            adapter_options[f"{name}_lora_a"] = adapters[index * 2]
            adapter_options[f"{name}_lora_b"] = adapters[index * 2 + 1]
            adapter_options[f"{name}_use_nora"] = True
    tuning = {}
    if args.tune:
        from flashrwkv2._internal.linear import _NoRA, linear, tune_linear

        source = shifted[0].detach().reshape(args.tokens, channels)
        tuning["dense"] = tune_linear(source, dense[0])
        first_weight = runtime_in[0].t() if operator == "pretrain" else rank_in[0]
        second_weight = runtime_out[0].t() if operator == "pretrain" else rank_out[0]
        tuning["lowrank_in"] = tune_linear(source, first_weight)
        reduced = linear(source, first_weight)
        tuning["lowrank_out"] = tune_linear(reduced, second_weight)
        if adapters:
            normalized = _NoRA.apply(adapters[0].detach())
            tuning["adapter_a"] = tune_linear(source, normalized)
            reduced = linear(source, normalized)
            tuning["adapter_b"] = tune_linear(
                reduced, adapters[1].detach().bfloat16(), torch.float32
            )
    variables = (
        *shifted,
        *dense,
        *runtime_in[:3],
        *runtime_out[:3],
        w0,
        kk,
        a0,
        ka,
        *adapters,
    )
    if first is not None:
        variables += (first, v0, runtime_in[3], runtime_out[3])
    gradient_inputs = tuple(t for t in variables if t.requires_grad)
    upstream = (
        [torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(8)]
        if backward
        else None
    )

    def run():
        if operator == "infer":
            outputs = infer_tmix_wkv_prepare_forward_varlen(
                *shifted,
                *dense,
                *rank_in,
                *rank_out,
                *vectors,
                v_first=first,
                head_size=args.head_size,
                **adapter_options,
            )
        else:
            outputs = pretrain_tmix_wkv_prepare_bf16(
                *shifted,
                *dense,
                w0,
                runtime_in[0],
                runtime_out[0],
                a0,
                runtime_in[1],
                runtime_out[1],
                runtime_in[2],
                runtime_out[2],
                kk,
                ka,
                v_first=first,
                v0=v0,
                v1=runtime_in[3],
                v2=runtime_out[3],
                head_size=args.head_size,
                **adapter_options,
            )
            if backward:
                torch.autograd.grad(outputs, gradient_inputs, upstream)
        return outputs

    sample = run()
    output_dtypes = [str(value.dtype).removeprefix("torch.") for value in sample]
    del sample
    return {
        "operator": "pretrain_tmix_wkv_prepare_bf16"
        if operator == "pretrain"
        else "infer_tmix_wkv_prepare_forward_varlen",
        "profile": f"tokens={args.tokens}/channels={channels}/rank={rank}/adapter_rank={args.adapter_rank}",
        "tokens": args.tokens,
        "channels": channels,
        "rank": rank,
        "head_size": args.head_size,
        "forward_backward": backward,
        "later_layer": args.later_layer,
        "implementation": args.implementation,
        "package_path": str(package_path),
        "seed": args.seed,
        "input_dtype": str(dtype).removeprefix("torch."),
        "output_dtypes": output_dtypes,
        "logical_inputs": "FP32 normal samples scaled by 0.03 and rounded through BF16 and FP16",
        "accumulation_dtype": "float32",
        "tuning": tuning,
        "steady_state": "stateless inputs and weights are reused",
        **measure_cuda(run, timing_mode=args.timing_mode),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--channels", type=int, default=4096)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--head-size", type=int, choices=(64, 128, 256), default=64)
    parser.add_argument("--adapter-rank", type=int, default=0)
    parser.add_argument("--operator", choices=("infer", "pretrain"))
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--later-layer", action="store_true")
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--seed", type=int, default=127)
    parser.add_argument(
        "--implementation", choices=("tilelang", "cuda-baseline"), default="tilelang"
    )
    parser.add_argument("--timing-mode", choices=("eager", "graph"), default="eager")
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="emit JSONL for model widths, M=1/27/256 and rank=32/64/128",
    )
    args = parser.parse_args()
    if args.timing_mode == "graph":
        torch.cuda.set_stream(torch.cuda.Stream())
    cases = [(args.tokens, args.channels, args.rank)]
    if args.sweep:
        cases = [
            (rows, channels, rank)
            for channels in (768, 1024, 2048, 2560, 4096, 5120)
            for rows in (1, 27, 256)
            for rank in (32, 64, 128)
        ]
    for tokens, channels, rank in cases:
        case = argparse.Namespace(
            **{**vars(args), "tokens": tokens, "channels": channels, "rank": rank}
        )
        print(json.dumps(benchmark(case)))


if __name__ == "__main__":
    main()
