# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from flashrwkv2.embedding import infer_embedding_ln0_forward_varlen


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.sm120
@pytest.mark.parametrize("channels", (8, 4096))
def test_embedding_ln0_packed(channels: int) -> None:
    torch.manual_seed(57)
    device = torch.device("cuda")
    embedding = torch.randn(5, channels, device=device, dtype=torch.bfloat16)
    weight = torch.randn(channels, device=device, dtype=torch.bfloat16)
    bias = torch.randn(channels, device=device, dtype=torch.bfloat16)
    output = infer_embedding_ln0_forward_varlen(embedding, weight, bias)
    expected = F.layer_norm(
        embedding.float(), (channels,), weight.float(), bias.float(), 1.0e-5
    ).half()
    assert output.dtype == torch.float16
    assert torch.allclose(output.float(), expected.float(), atol=0.05, rtol=0.05)


def test_embedding_rejects_cpu_before_native_launch() -> None:
    embedding = torch.zeros(2, 8, dtype=torch.bfloat16)
    affine = torch.zeros(8, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="CUDA"):
        infer_embedding_ln0_forward_varlen(embedding, affine, affine)
