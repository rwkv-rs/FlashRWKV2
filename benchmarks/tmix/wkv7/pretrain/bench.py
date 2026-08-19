# SPDX-License-Identifier: MIT

from __future__ import annotations

import json

import torch
from _timing import measure_cuda

from flashrwkv2 import pretrain_tmix_wkv7_recurrent_bf16
from flashrwkv2.tmix.wkv7 import _extension

SOURCE_REVISION = "952102498e9ed367ea0a59ee64106916d474d30f"
NATIVE_SYMBOLS = (
    "pretrain_tmix_wkv7_recurrent_forward",
    "pretrain_tmix_wkv7_recurrent_backward",
)


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
        "source_revision": SOURCE_REVISION,
        "operator": "pretrain_tmix_wkv7_recurrent_bf16",
        "native_symbols": NATIVE_SYMBOLS,
        "native_symbols_loaded": all(
            hasattr(_extension(), symbol) for symbol in NATIVE_SYMBOLS
        ),
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
    print(json.dumps(run(), indent=2))
