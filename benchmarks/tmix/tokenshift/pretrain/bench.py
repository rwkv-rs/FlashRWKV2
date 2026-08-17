# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import json
import time

import torch

from flashrwkv2.tmix.tokenshift import (
    pretrain_tmix_tokenshift_bf16,
    statetune_tmix_tokenshift_bf16,
)


def torch_stateful(x, initial_shift, params):
    previous = torch.cat((initial_shift[:, None, :], x[:, :-1, :]), dim=1)
    return tuple(x + (previous - x) * parameter for parameter in params) + (
        x[:, -1, :].contiguous(),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seqlen", type=int, default=32)
    parser.add_argument("--channels", type=int, default=4096)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--measurements", type=int, default=3)
    parser.add_argument(
        "--operator", choices=("pretrain", "stateful", "torch"), default="pretrain"
    )
    parser.add_argument("--backward", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda")
    x = torch.zeros(
        args.batch,
        args.seqlen,
        args.channels,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=args.backward,
    )
    initial_shift = torch.zeros(
        args.batch,
        args.channels,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=args.backward,
    )
    params = [
        torch.zeros(
            args.channels,
            device=device,
            dtype=torch.bfloat16,
            requires_grad=args.backward,
        )
        for _ in range(6)
    ]

    def run():
        if args.operator == "pretrain":
            outputs = pretrain_tmix_tokenshift_bf16(x, *params)
        elif args.operator == "stateful":
            outputs = statetune_tmix_tokenshift_bf16(x, initial_shift, *params)
        else:
            outputs = torch_stateful(x, initial_shift, params)
        if args.backward:
            sum(output.sum() for output in outputs).backward()
            for tensor in (x, initial_shift, *params):
                tensor.grad = None

    for _ in range(10):
        run()
    torch.cuda.synchronize()
    samples = []
    for _ in range(args.measurements):
        start = time.perf_counter()
        for _ in range(args.iters):
            run()
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
                "measurements_us": samples,
                "mean_us": sum(samples) / len(samples),
                "gpu": torch.cuda.get_device_name(),
            }
        )
    )


if __name__ == "__main__":
    main()
