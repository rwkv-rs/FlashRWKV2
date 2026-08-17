# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import json
import time

import torch

from flashrwkv2.cmix import (
    infer_cmix_forward_varlen,
    pretrain_cmix_bf16,
    statetune_cmix_bf16,
)


def torch_stateful(x, initial_shift, x_k, key, value):
    previous = torch.cat((initial_shift[:, None, :], x[:, :-1, :]), dim=1)
    mixed = x + (previous - x) * x_k
    return torch.relu(mixed @ key.t()).square() @ value.t(), x[:, -1].contiguous()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seqlen", type=int, default=32)
    parser.add_argument("--channels", type=int, default=4096)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--measurements", type=int, default=3)
    parser.add_argument(
        "--operator",
        choices=("infer", "pretrain", "stateful", "torch"),
        default="infer",
    )
    parser.add_argument("--backward", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda")
    if args.operator == "infer" and args.backward:
        raise ValueError("the inference benchmark does not support --backward")
    dtype = torch.float16 if args.operator == "infer" else torch.bfloat16
    x = torch.zeros(
        args.batch,
        args.seqlen,
        args.channels,
        device=device,
        dtype=dtype,
        requires_grad=args.backward,
    )
    initial_shift = torch.zeros(
        args.batch,
        args.channels,
        device=device,
        dtype=dtype,
        requires_grad=args.backward,
    )
    x_k = torch.zeros(
        args.channels, device=device, dtype=dtype, requires_grad=args.backward
    )
    key = torch.zeros(
        4 * args.channels,
        args.channels,
        device=device,
        dtype=dtype,
        requires_grad=args.backward,
    )
    value = torch.zeros(
        args.channels,
        4 * args.channels,
        device=device,
        dtype=dtype,
        requires_grad=args.backward,
    )

    if args.operator == "infer":
        rows = args.batch * args.seqlen
        x = x.reshape(rows, args.channels)
        res = torch.zeros_like(x)
        weight = torch.ones(args.channels, device=device, dtype=dtype)
        bias = torch.zeros_like(weight)
        key = torch.zeros(args.channels, args.channels, device=device, dtype=dtype)
        value = torch.zeros_like(key)
        shift_state = initial_shift.detach().clone()
        cu_seqlens = torch.arange(
            0,
            rows + 1,
            args.seqlen,
            device=device,
            dtype=torch.int32,
        )
        state_indices = torch.arange(args.batch, device=device, dtype=torch.int32)

    def run():
        if args.operator == "infer":
            outputs = infer_cmix_forward_varlen(
                x,
                res,
                weight,
                bias,
                x_k,
                key,
                value,
                shift_state_pool=shift_state,
                cu_seqlens=cu_seqlens,
                state_indices=state_indices,
                max_seqlen=args.seqlen,
            )
        elif args.operator == "pretrain":
            outputs = (pretrain_cmix_bf16(x, x_k, key, value),)
        elif args.operator == "stateful":
            outputs = statetune_cmix_bf16(x, initial_shift, x_k, key, value)
        else:
            outputs = torch_stateful(x, initial_shift, x_k, key, value)
        if args.backward:
            sum(output.sum() for output in outputs).backward()
            for tensor in (x, initial_shift, x_k, key, value):
                tensor.grad = None
        return outputs

    for _ in range(5):
        outputs = run()
    torch.cuda.synchronize()
    samples = []
    for _ in range(args.measurements):
        start = time.perf_counter()
        for _ in range(args.iters):
            outputs = run()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e6 / args.iters)
    print(
        json.dumps(
            {
                "operator": args.operator,
                "forward_backward": args.backward,
                "batch": args.batch,
                "seqlen": args.seqlen,
                "channels": args.channels,
                "raw_latency_us": samples,
                "mean_us": sum(samples) / len(samples),
                "gpu": torch.cuda.get_device_name(),
                "correctness": (
                    "passed"
                    if all(torch.isfinite(output).all() for output in outputs)
                    else "failed"
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
