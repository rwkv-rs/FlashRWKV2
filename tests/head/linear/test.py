# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest
import torch

from flashrwkv2.head.linear import (
    infer_head_linear_all_forward_varlen,
    infer_head_linear_last_forward_varlen,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.sm120
@pytest.mark.memcheck
@pytest.mark.parametrize(
    ("rows", "channels", "vocab"), ((4, 8, 11), (1, 4096, 65536))
)
def test_head_all_and_last_linear(
    rows: int, channels: int, vocab: int
) -> None:
    torch.manual_seed(59)
    device = torch.device("cuda")
    x = torch.randn(rows, channels, device=device, dtype=torch.float16)
    weight = torch.randn(vocab, channels, device=device, dtype=torch.float16)
    all_logits = infer_head_linear_all_forward_varlen(x, weight)
    assert torch.allclose(all_logits.float(), x.float() @ weight.float().t(), atol=0.04, rtol=0.04)
    last_logits = infer_head_linear_last_forward_varlen(
        x, weight, tokens_count=max(rows, 2)
    )
    assert torch.allclose(last_logits.float(), x.float() @ weight.float().t(), atol=0.04, rtol=0.04)
    with pytest.raises(ValueError, match="positive"):
        infer_head_linear_last_forward_varlen(x, weight, tokens_count=0)


def test_head_linear_rejects_cpu_before_native_launch() -> None:
    x = torch.zeros(2, 8, dtype=torch.float16)
    weight = torch.zeros(11, 8, dtype=torch.float16)
    with pytest.raises(ValueError, match="CUDA"):
        infer_head_linear_all_forward_varlen(x, weight)
