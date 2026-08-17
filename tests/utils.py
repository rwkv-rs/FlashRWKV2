"""Shared assertions for observable FlashRWKV2 test behavior."""

# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

_TOLERANCES = json.loads(
    (Path(__file__).parent / "fixtures/tolerances-v1.json").read_text(
        encoding="utf-8"
    )
)


def require_cuda_backend(
    expected_backend: str,
    compatible_major: int,
    *required_symbols: str,
) -> Any:
    """Return the selected backend, skipping only unavailable hardware."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    capability = tuple(torch.cuda.get_device_capability())
    if capability[0] != compatible_major:
        pytest.skip(
            f"{expected_backend} requires compute capability major "
            f"{compatible_major}, found sm{capability[0]}{capability[1]}"
        )

    import flashrwkv2

    backend = flashrwkv2._C
    assert backend is not None, "matching CUDA hardware did not load a backend"
    assert backend.__name__ == f"flashrwkv2.{expected_backend}"
    missing = [name for name in required_symbols if not hasattr(backend, name)]
    assert not missing, f"{expected_backend} is missing required symbols: {missing}"
    return backend


def assert_tensor_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    profile: str | None = None,
    field: str = "output",
    atol: float | None = None,
    rtol: float | None = None,
) -> None:
    """Compare tensors with an explicit or repository-calibrated tolerance."""
    if profile is not None:
        contract = _TOLERANCES[profile]
        if atol is None:
            atol = contract.get(f"{field}_atol")
        if rtol is None:
            rtol = contract.get(f"{field}_relative_rmse")
    if atol is None:
        atol = 0.0
    if rtol is None:
        rtol = 0.0
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def assert_state_unchanged(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Assert exact preservation of caller-owned or inactive state."""
    assert torch.equal(actual, expected), "state changed outside the selected update"


def assert_finite_positive(value: float | torch.Tensor) -> None:
    """Assert that a scalar sample is finite and strictly positive."""
    scalar = torch.as_tensor(value)
    assert scalar.numel() == 1
    assert torch.isfinite(scalar).item()
    assert scalar.item() > 0
