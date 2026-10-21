# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from flashrwkv2.embedding import infer_embedding_ln0_forward_varlen


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.cuda
@pytest.mark.parametrize(
    "channels",
    (
        8,
        pytest.param(
            4096,
            marks=(pytest.mark.memcheck, pytest.mark.racecheck),
        ),
    ),
)
def test_embedding_ln0_packed(channels: int) -> None:
    torch.manual_seed(57)
    device = torch.device("cuda")
    embedding = torch.randn(
        5, channels, device=device, dtype=torch.bfloat16
    ).requires_grad_()
    weight = torch.randn(channels, device=device, dtype=torch.bfloat16).requires_grad_()
    bias = torch.randn(channels, device=device, dtype=torch.bfloat16).requires_grad_()
    output = infer_embedding_ln0_forward_varlen(embedding, weight, bias)
    refs = [
        value.detach().cpu().requires_grad_() for value in (embedding, weight, bias)
    ]
    expected = F.layer_norm(
        refs[0].float(), (channels,), refs[1].float(), refs[2].float(), 1.0e-5
    ).bfloat16()
    assert output.dtype == torch.bfloat16
    assert torch.allclose(output.float().cpu(), expected.float(), atol=0.05, rtol=0.05)
    upstream = torch.randn_like(output)
    gradients = torch.autograd.grad(output, (embedding, weight, bias), upstream)
    reference_gradients = torch.autograd.grad(expected, refs, upstream.cpu())
    for actual, reference in zip(gradients, reference_gradients, strict=True):
        torch.testing.assert_close(actual.cpu(), reference, atol=0.05, rtol=0.05)
    with torch.no_grad():
        inference = infer_embedding_ln0_forward_varlen(embedding, weight, bias)
    assert torch.equal(output, inference)


def test_embedding_rejects_cpu_before_launch() -> None:
    embedding = torch.zeros(2, 8, dtype=torch.bfloat16)
    affine = torch.zeros(8, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="CUDA"):
        infer_embedding_ln0_forward_varlen(embedding, affine, affine)
