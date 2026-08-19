# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest
import torch
from torch.nn import functional

from flashrwkv2.loss.l2wrap_ce import pretrain_l2wrap_ce_bf16


def _inputs(
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(731)
    logits = torch.randn(2, 3, 16, device=device, dtype=dtype).mul_(0.25)
    targets = torch.tensor(
        [[0, 7, 15], [3, 9, 1]],
        device=device,
        dtype=torch.int64,
    )
    return logits, targets


def _expected_gradient(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    rows = targets.numel()
    flattened = logits.float().reshape(rows, -1)
    gradient = torch.softmax(flattened, dim=-1)
    row_indices = torch.arange(rows, device=logits.device)
    gradient[row_indices, targets.reshape(-1)] -= 1
    gradient /= rows
    max_values, argmax = flattened.max(dim=-1)
    gradient[row_indices, argmax] += max_values * (1.0e-4 / rows)
    return gradient.reshape_as(logits).to(logits.dtype)


def test_l2wrap_rejects_non_cuda_inputs_before_native_launch() -> None:
    with pytest.raises(ValueError, match="contiguous CUDA"):
        pretrain_l2wrap_ce_bf16(*_inputs(device="cpu"))


def test_l2wrap_rejects_unsupported_logits_dtype() -> None:
    with pytest.raises(TypeError, match=r"torch\.bfloat16 or torch\.float32"):
        pretrain_l2wrap_ce_bf16(*_inputs(device="cpu", dtype=torch.float16))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.sm90
@pytest.mark.sm120
@pytest.mark.memcheck
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float32))
def test_l2wrap_forward_backward_matches_train_temp_contract(
    dtype: torch.dtype,
) -> None:
    logits, targets = _inputs(device="cuda", dtype=dtype)
    logits.requires_grad_(True)

    loss = pretrain_l2wrap_ce_bf16(logits, targets)
    reference_loss = functional.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
    )
    loss.backward()

    assert loss.dtype == torch.float32
    torch.testing.assert_close(loss, reference_loss, atol=2.0e-4, rtol=2.0e-4)
    assert logits.grad is not None
    torch.testing.assert_close(
        logits.grad,
        _expected_gradient(logits.detach(), targets),
        atol=0.002 if dtype == torch.bfloat16 else 2.0e-5,
        rtol=0.01,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.sm90
@pytest.mark.sm120
def test_l2wrap_rejects_target_shape_and_range_without_launching() -> None:
    logits, targets = _inputs(device="cuda")
    with pytest.raises(ValueError, match="one entry"):
        pretrain_l2wrap_ce_bf16(logits, targets.reshape(-1)[:-1])

    invalid_targets = targets.clone()
    invalid_targets[0, 0] = logits.shape[-1]
    with pytest.raises(ValueError, match=r"\[0,vocab\)"):
        pretrain_l2wrap_ce_bf16(logits, invalid_targets)
