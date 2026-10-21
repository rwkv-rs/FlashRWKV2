# SPDX-License-Identifier: MIT

from __future__ import annotations

import json

import torch
from _timing import measure_cuda

from flashrwkv2 import pretrain_tmix_wkv7_recurrent_bf16


def run() -> dict[str, object]:
    batch, tokens, heads, head_size = 2, 512, 12, 64
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
        output = pretrain_tmix_wkv7_recurrent_bf16(*values)
        torch.autograd.grad(output, values, grad_output, retain_graph=False)

    timing = measure_cuda(step)
    device = torch.cuda.current_device()
    return {
        "operator": "pretrain_tmix_wkv7_recurrent_bf16",
        "gpu": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "cuda": torch.version.cuda,
        "shape": {
            "B": batch,
            "T": tokens,
            "H": heads,
            "N": head_size,
        },
        "workload": "forward+backward",
        "steady_state": "autograd graph is recreated and released each launch",
        "timing": timing,
        "tokens_per_second_mean": batch * tokens * 1_000_000.0 / timing["mean_us"],
    }


if __name__ == "__main__":
    print(json.dumps(run()))
