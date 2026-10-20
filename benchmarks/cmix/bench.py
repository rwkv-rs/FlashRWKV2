# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import json

import torch
from _timing import measure_cuda

from flashrwkv2._internal.linear import _NoRA, linear, tune_linear
from flashrwkv2.cmix import (
    infer_cmix_forward_varlen,
    pretrain_cmix_bf16,
    statetune_cmix_bf16,
)
from flashrwkv2.tmix.wkv7 import prepare_tmix_wkv7_recurrent_metadata


def _torch_linear(x, weight, a=None, b=None):
    output = x @ weight.t()
    if a is not None:
        normalized = (
            a.float() / (a.float().norm(dim=0, keepdim=True) + 1.0e-6)
        ).bfloat16()
        features = x @ normalized.t()
        output = (
            output.float() + features.float() @ b.bfloat16().float().t()
        ).bfloat16()
    return output


def benchmark(args):
    if args.operator == "infer" and args.backward:
        raise ValueError("the inference benchmark does not support --backward")
    if args.operator == "torch" and args.tune:
        raise ValueError("--tune applies to TileLang operators")
    channels = args.channels
    features = 4 * channels if args.features is None else args.features
    rows = args.batch * args.seqlen
    torch.manual_seed(113)

    def random(*shape):
        return (
            torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.02
        ).requires_grad_(args.backward)

    x, initial, mix = (
        random(args.batch, args.seqlen, channels),
        random(args.batch, channels),
        random(channels),
    )
    key, value = random(features, channels), random(channels, features)
    adapters, adapter_options = [], {}
    if args.adapter_rank:
        key.requires_grad_(False)
        value.requires_grad_(False)
        adapters = [
            (torch.randn(shape, device="cuda") * 0.01).requires_grad_(args.backward)
            for shape in (
                (args.adapter_rank, channels),
                (features, args.adapter_rank),
                (args.adapter_rank, features),
                (channels, args.adapter_rank),
            )
        ]
        adapter_options = dict(
            zip(
                ("key_lora_a", "key_lora_b", "value_lora_a", "value_lora_b"),
                adapters,
                strict=True,
            )
        )
        adapter_options.update(key_use_nora=True, value_use_nora=True)
    if args.operator == "infer":
        x = x.reshape(rows, channels)
        value = value.t().contiguous()
        res = torch.zeros_like(x)
        weight = torch.ones(channels, device="cuda", dtype=torch.bfloat16)
        bias = torch.zeros_like(weight)
        state = initial.clone()
        cu = (
            torch.arange(args.batch + 1, device="cuda", dtype=torch.int32) * args.seqlen
        )
        slots = torch.arange(args.batch, device="cuda", dtype=torch.int32)
        ticket = prepare_tmix_wkv7_recurrent_metadata(
            cu,
            slots,
            state_pool_size=args.batch,
            total_tokens=rows,
            max_seqlen=args.seqlen,
        )
    tuning = {}
    if args.tune:
        source = x.detach().reshape(rows, channels)
        tuning["key"] = tune_linear(source, key)
        hidden = linear(source, key).float().relu().square().bfloat16()
        tuning["value"] = tune_linear(
            hidden, value.t() if args.operator == "infer" else value
        )
        if adapters:
            for name, adapter_source, a, b in (
                ("key", source, *adapters[:2]),
                ("value", hidden, *adapters[2:]),
            ):
                normalized = _NoRA.apply(a.detach())
                tuning[f"{name}_adapter_a"] = tune_linear(adapter_source, normalized)
                projected = linear(adapter_source, normalized)
                tuning[f"{name}_adapter_b"] = tune_linear(
                    projected, b.detach().bfloat16(), torch.float32
                )

    variables = (x, mix, key, value, *adapters)
    if args.operator in ("stateful", "torch"):
        variables += (initial,)
    gradient_inputs = tuple(t for t in variables if t.requires_grad)

    def run():
        if args.operator == "infer":
            output = infer_cmix_forward_varlen(
                x,
                res,
                weight,
                bias,
                mix,
                key,
                value,
                shift_state_pool=state,
                cu_seqlens=cu,
                state_indices=slots,
                validated_metadata=ticket,
                **adapter_options,
            )
        elif args.operator == "pretrain":
            output = (pretrain_cmix_bf16(x, mix, key, value, **adapter_options),)
        elif args.operator == "stateful":
            output = statetune_cmix_bf16(x, initial, mix, key, value, **adapter_options)
        else:
            previous = torch.cat((initial[:, None], x[:, :-1]), dim=1)
            mixed = (
                x.float() + (previous.float() - x.float()) * mix.float()
            ).bfloat16()
            hidden = (
                _torch_linear(mixed, key, *adapters[:2])
                .float()
                .relu()
                .square()
                .bfloat16()
            )
            output = (
                _torch_linear(hidden, value, *adapters[2:]),
                x[:, -1].contiguous(),
            )
        if args.backward:
            torch.autograd.grad(sum(t.float().sum() for t in output), gradient_inputs)
        return output

    timing = measure_cuda(
        run,
        before_batch=(lambda: state.copy_(initial))
        if args.operator == "infer"
        else None,
    )
    return {
        "operator": args.operator,
        "forward_backward": args.backward,
        "batch": args.batch,
        "seqlen": args.seqlen,
        "channels": channels,
        "features": features,
        "adapter_rank": args.adapter_rank,
        "input_dtype": "bfloat16",
        "accumulation_dtype": "float32",
        "tuning": tuning,
        "profile": f"operator={args.operator}/backward={args.backward}/batch={args.batch}/seqlen={args.seqlen}/channels={channels}/features={features}/adapter_rank={args.adapter_rank}",
        "steady_state": "shift state is reset before each timing batch"
        if args.operator == "infer"
        else "autograd graphs are recreated each launch"
        if args.backward
        else "stateless inputs are reused",
        **timing,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seqlen", type=int, default=32)
    parser.add_argument("--channels", type=int, default=4096)
    parser.add_argument(
        "--features", type=int, help="intermediate width; defaults to 4*C"
    )
    parser.add_argument(
        "--operator",
        choices=("infer", "pretrain", "stateful", "torch"),
        default="infer",
    )
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--adapter-rank", type=int, default=0)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="emit JSONL over model widths and decode/prefill batches",
    )
    args = parser.parse_args()
    cases = [(args.batch, args.seqlen, args.channels)]
    if args.sweep:
        cases = [
            (batch, length, channels)
            for channels in (768, 1024, 2048, 2560, 4096, 5120)
            for batch, length in ((1, 1), (1, 27), (4, 64))
        ]
    for batch, seqlen, channels in cases:
        case = argparse.Namespace(
            **{**vars(args), "batch": batch, "seqlen": seqlen, "channels": channels}
        )
        print(json.dumps(benchmark(case)))


if __name__ == "__main__":
    main()
