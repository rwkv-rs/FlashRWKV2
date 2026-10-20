# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import json

import torch
from _timing import measure_cuda

from flashrwkv2.head.l2wrap_ce import pretrain_head_l2wrap_ce_bf16


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--adapter-rank", type=int, default=0)
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--chunk-rows", type=int, default=4096)
    parser.add_argument(
        "--train-base",
        action="store_true",
        help="include base-weight VJP in --backward; otherwise freeze the base weight",
    )
    parser.add_argument("--timing-mode", choices=("eager", "graph"), default="eager")
    parser.add_argument("--seed", type=int, default=113)
    args = parser.parse_args()
    if args.timing_mode == "graph":
        parser.error(
            "HeadL2Wrap validates targets with .item() and requires eager timing"
        )
    if args.train_base and not args.backward:
        parser.error("--train-base requires --backward")
    if args.chunk_rows <= 0:
        parser.error("--chunk-rows must be positive")
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    hidden = torch.randn(
        args.batch,
        args.tokens,
        args.channels,
        device=device,
        dtype=torch.bfloat16,
    ).requires_grad_(args.backward)
    weight = torch.randn(
        65536,
        args.channels,
        device=device,
        dtype=torch.bfloat16,
    ).requires_grad_(args.backward and args.train_base)
    targets = torch.randint(
        65536, (args.batch * args.tokens,), device=device, dtype=torch.int64
    )
    options = {"chunk_rows": args.chunk_rows}
    inputs = [hidden] if args.backward else []
    if weight.requires_grad:
        inputs.append(weight)
    if args.adapter_rank:
        a = (
            torch.randn(args.adapter_rank, args.channels, device=device) * 0.01
        ).requires_grad_(args.backward)
        b = (
            torch.randn(65536, args.adapter_rank, device=device) * 0.01
        ).requires_grad_(args.backward)
        options.update(lora_a=a, lora_b=b, use_nora=True)
        if args.backward:
            inputs.extend((a, b))

    def run():
        loss = pretrain_head_l2wrap_ce_bf16(hidden, weight, targets, **options)
        if args.backward:
            torch.autograd.grad(loss, inputs)
        return loss

    # Compile and allocate once before measuring the memory of one operation.
    sample = run()
    loss_dtype = str(sample.dtype).removeprefix("torch.")
    del sample
    torch.cuda.synchronize()
    baseline_bytes = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    run()
    torch.cuda.synchronize()
    peak_bytes = torch.cuda.max_memory_allocated()
    memory = {
        "scope": "one forward/backward call after warmup"
        if args.backward
        else "one forward call after warmup",
        "baseline_allocated_bytes": baseline_bytes,
        "peak_allocated_bytes": peak_bytes,
        "peak_increment_bytes": peak_bytes - baseline_bytes,
    }
    timing = measure_cuda(run, timing_mode=args.timing_mode)
    print(
        json.dumps(
            {
                "operator": "pretrain_head_l2wrap_ce_bf16",
                "seed": args.seed,
                "forward_backward": args.backward,
                "chunk_rows": args.chunk_rows,
                "base_weight": "trainable" if args.train_base else "frozen",
                "input_dtype": str(hidden.dtype).removeprefix("torch."),
                "base_weight_dtype": str(weight.dtype).removeprefix("torch."),
                "loss_dtype": loss_dtype,
                "memory": memory,
                "profile": (
                    f"batch={args.batch}/tokens={args.tokens}/channels={args.channels}/adapter_rank={args.adapter_rank}/backward={args.backward}/train_base={args.train_base}/chunk_rows={args.chunk_rows}"
                ),
                "steady_state": "autograd graphs are recreated for each forward/backward call"
                if args.backward
                else "forward outputs are discarded after each launch",
                **timing,
            }
        )
    )


if __name__ == "__main__":
    main()
