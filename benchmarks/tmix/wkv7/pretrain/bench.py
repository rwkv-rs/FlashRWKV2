# SPDX-License-Identifier: MIT

from __future__ import annotations

import json

import torch

from flashrwkv2 import pretrain_tmix_wkv7_recurrent_bf16


SOURCE_REVISION = "952102498e9ed367ea0a59ee64106916d474d30f"
NATIVE_NAMESPACE = "rwkv7_clampw_v3"


def run(*, warmup: int = 5, iterations: int = 50) -> dict[str, object]:
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

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        step()
    end.record()
    end.synchronize()
    latency_ms = start.elapsed_time(end) / iterations
    device = torch.cuda.current_device()
    return {
        "source_revision": SOURCE_REVISION,
        "operator": "pretrain_tmix_wkv7_recurrent_bf16",
        "native_namespace": NATIVE_NAMESPACE,
        "native_namespace_loaded": hasattr(torch.ops.rwkv7_clampw_v3, "forward"),
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
        "latency_ms": latency_ms,
        "tokens_per_second": batch * tokens / (latency_ms / 1000.0),
        "warmup": warmup,
        "iterations": iterations,
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
