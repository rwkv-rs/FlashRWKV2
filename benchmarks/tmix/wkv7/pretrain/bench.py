# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import json

import torch
from _timing import measure_cuda

from flashrwkv2 import pretrain_tmix_wkv7_recurrent_bf16


def run() -> dict[str, object]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--timing-mode", choices=("eager", "graph"), default="eager")
    parser.add_argument("--backward-only", action="store_true")
    args = parser.parse_args()
    batch, tokens, heads, head_size = args.batch, args.tokens, args.heads, 64
    shape = (batch, tokens, heads * head_size)
    torch.manual_seed(20260806)
    values = tuple(
        (torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.05)
        .contiguous()
        .requires_grad_()
        for _ in range(6)
    )
    grad_output = torch.randn(shape, device="cuda", dtype=torch.bfloat16)


    def step() -> None:
        result = output if args.backward_only else pretrain_tmix_wkv7_recurrent_bf16(*values)
        torch.autograd.grad(result, values, grad_output, retain_graph=args.backward_only)

    timing_counts = {"warmup": 5, "launches_per_batch": 1} if args.backward_only or args.timing_mode == "graph" else {}
    stream = torch.cuda.current_stream()
    if args.timing_mode == "graph":
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        output = pretrain_tmix_wkv7_recurrent_bf16(*values) if args.backward_only else None
        timing = measure_cuda(step, timing_mode=args.timing_mode, **timing_counts)
    device = torch.cuda.current_device()
    return {
        "operator": "pretrain_tmix_wkv7_recurrent_bf16",
        "gpu": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
        "shape": {
            "B": batch,
            "T": tokens,
            "H": heads,
            "N": head_size,
        },
        "workload": "backward" if args.backward_only else "forward+backward",
        "steady_state": "reuse saved forward" if args.backward_only else "autograd graph is recreated and released each launch",
        "timing": timing,
        "tokens_per_second_mean": batch * tokens * 1_000_000.0 / timing["mean_us"],
    }


if __name__ == "__main__":
    print(json.dumps(run()))
