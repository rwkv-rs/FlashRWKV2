# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from _timing import measure_cuda

import flashrwkv2
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
    package_path = Path(flashrwkv2.__file__).resolve()
    is_tilelang = (package_path.parent / "_internal" / "linear.py").is_file()
    if is_tilelang != (args.implementation == "tilelang"):
        raise ValueError(
            "--implementation does not match the package selected by PYTHONPATH"
        )
    if args.operator == "infer" and args.backward:
        raise ValueError("the inference benchmark does not support --backward")
    if args.operator == "torch" and args.tune:
        raise ValueError("--tune applies to TileLang operators")
    if args.implementation == "cuda-baseline":
        if args.tune or args.operator == "torch":
            raise ValueError("the CUDA baseline uses its own operators without tuning")
        if args.operator == "infer" and args.adapter_rank:
            raise ValueError(
                "the CUDA baseline inference API has no adapter parameters"
            )
    channels = args.channels
    features = 4 * channels if args.features is None else args.features
    if (
        args.implementation == "cuda-baseline"
        and args.operator != "infer"
        and features != 4 * channels
    ):
        raise ValueError("CUDA baseline training requires features=4*channels")
    rows = args.batch * args.seqlen
    torch.manual_seed(args.seed)
    dtype = (
        torch.float16
        if args.implementation == "cuda-baseline" and args.operator == "infer"
        else torch.bfloat16
    )

    def random(*shape):
        return (
            (torch.randn(shape, device="cuda") * 0.02).bfloat16().half().to(dtype)
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
        res = random(rows, channels)
        weight = torch.ones(channels, device="cuda", dtype=dtype)
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
        from flashrwkv2._internal.elementwise import activation
        from flashrwkv2._internal.linear import (
            _NoRA,
            adapter_linear,
            linear,
            tune_linear,
        )
        from flashrwkv2.post_norm.tilelang import norm, tune_norm
        from flashrwkv2.tmix.tokenshift.tilelang import (
            fused_post_norm_tokenshift,
            tokenshift,
            tune_fused_tokenshift,
            tune_tokenshift,
        )

        with torch.no_grad():
            source = x.detach().reshape(rows, channels)
            if args.operator == "infer" and args.seqlen == 1:
                tuning["fused_tokenshift"] = tune_fused_tokenshift(
                    source,
                    res,
                    weight,
                    bias,
                    (mix,),
                    state,
                    ticket._state_indices_snapshot(),
                    ticket._status(),
                )
                _, mixed = fused_post_norm_tokenshift(
                    source,
                    res,
                    weight,
                    bias,
                    (mix,),
                    state.clone(),
                    ticket._state_indices_snapshot(),
                    ticket._status(),
                )
                source = mixed[0]
            else:
                status = None
                initial_shift = initial if args.operator != "pretrain" else None
                tune_cu = (
                    torch.arange(args.batch + 1, device="cuda", dtype=torch.int32)
                    * args.seqlen
                )
                if args.operator == "infer":
                    tuning["norm"] = tune_norm(source, weight, bias, res=res)
                    source = norm(source, weight, bias, res=res)
                    tune_cu, status = (
                        ticket._query_start_loc_snapshot(),
                        ticket._status(),
                    )
                tuning["tokenshift"] = tune_tokenshift(
                    source,
                    (mix,),
                    tune_cu,
                    initial_shift,
                    _status=status,
                )
                mixed, _ = tokenshift(
                    source, (mix,), tune_cu, initial_shift, _status=status
                )
                source = mixed[0]
            tuning["key"] = tune_linear(source, key)
            projected = (
                adapter_linear(
                    source,
                    key,
                    *adapters[:2],
                    use_nora=True,
                )
                if adapters
                else linear(source, key)
            )
            hidden = activation(projected, "relu2")
            tuning["value"] = tune_linear(
                hidden, value.t() if args.operator == "infer" else value
            )
            if adapters:
                for name, adapter_source, a, b in (
                    ("key", source, *adapters[:2]),
                    ("value", hidden, *adapters[2:]),
                ):
                    normalized = _NoRA.apply(a.detach())
                    tuning[f"{name}_adapter_a"] = tune_linear(
                        adapter_source, normalized
                    )
                    features_a = linear(adapter_source, normalized)
                    tuning[f"{name}_adapter_b"] = tune_linear(
                        features_a, b.detach().bfloat16(), torch.float32
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
                max_seqlen=args.seqlen,
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

    sample = run()
    output_dtypes = [str(value.dtype).removeprefix("torch.") for value in sample]
    del sample
    if args.tune and not args.backward:
        tuning["key"] = tune_linear(source, key, benchmark=run)
        tuning["value"] = tune_linear(
            hidden, value.t() if args.operator == "infer" else value, benchmark=run
        )
    timing = measure_cuda(
        run,
        before_batch=(lambda: state.copy_(initial))
        if args.operator == "infer"
        else None,
        timing_mode=args.timing_mode,
    )
    return {
        "operator": args.operator,
        "implementation": args.implementation,
        "package_path": str(package_path),
        "seed": args.seed,
        "forward_backward": args.backward,
        "batch": args.batch,
        "seqlen": args.seqlen,
        "channels": channels,
        "features": features,
        "adapter_rank": args.adapter_rank,
        "input_dtype": str(x.dtype).removeprefix("torch."),
        "output_dtypes": output_dtypes,
        "key_weight_dtype": str(key.dtype).removeprefix("torch."),
        "value_weight_dtype": str(value.dtype).removeprefix("torch."),
        "shift_state_dtype": str(initial.dtype).removeprefix("torch."),
        "accumulation_dtype": "float32",
        "logical_inputs": "FP32 normal samples scaled by 0.02, rounded through BF16 and FP16; identical values in both formats",
        "metadata": {
            "total_tokens": rows,
            "sequences": args.batch,
            "state_pool_size": args.batch,
            "max_seqlen": args.seqlen,
            "preparation": "outside timing",
        }
        if args.operator == "infer"
        else None,
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
        "--implementation",
        choices=("tilelang", "cuda-baseline"),
        default="tilelang",
        help="select TileLang or the saved CUDA baseline package through PYTHONPATH",
    )
    parser.add_argument("--timing-mode", choices=("eager", "graph"), default="eager")
    parser.add_argument("--seed", type=int, default=113)
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="emit JSONL over model widths and decode/prefill batches",
    )
    args = parser.parse_args()
    if args.timing_mode == "graph":
        torch.cuda.set_stream(torch.cuda.Stream())
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
