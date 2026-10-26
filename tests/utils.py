"""Shared assertions for observable FlashRWKV2 test behavior."""

# SPDX-License-Identifier: MIT

from __future__ import annotations

import torch
from torch.utils._python_dispatch import TorchDispatchMode

class TorchOperationRecorder(TorchDispatchMode):
    """Record ATen calls without initializing a CUDA profiler."""

    def __init__(self) -> None:
        super().__init__()
        self.operations: list[str] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.operations.append(func._schema.name)
        return func(*args, **(kwargs or {}))


def nora_linear_reference(x, weight, a, b, scale=1.0, use_nora=True):
    """PEFT NoRA expression with FP32 norm reduction and BF16 GEMMs."""
    normalized = a.float()
    if use_nora:
        normalized = normalized / (normalized.norm(dim=0, keepdim=True) + 1.0e-6)
    normalized = normalized.to(x.dtype)
    return x @ weight.t() + ((x @ normalized.t()) @ b.to(x.dtype).t()) * scale


def assert_relative_rmse(actual, expected, tolerance=0.025):
    error = (actual.float() - expected.float()).square().mean().sqrt()
    magnitude = expected.float().square().mean().sqrt().clamp_min(1.0e-8)
    assert (error / magnitude).item() < tolerance, (error.item(), magnitude.item())
