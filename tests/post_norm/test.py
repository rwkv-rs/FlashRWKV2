# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from flashrwkv2.post_norm import infer_post_norm_output_forward_varlen


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.cuda
@pytest.mark.memcheck
@pytest.mark.parametrize("channels", (8, 4096))
def test_post_norm_output(channels: int) -> None:
    torch.manual_seed(53)
    device = torch.device("cuda")
    x = torch.randn(5, channels, device=device, dtype=torch.float16)
    res = torch.randn_like(x)
    weight = torch.randn(channels, device=device, dtype=torch.float16)
    bias = torch.randn(channels, device=device, dtype=torch.float16)
    expected_sum = (x + res).float()
    last = infer_post_norm_output_forward_varlen(x, res, weight, bias)
    assert torch.allclose(
        last.float(),
        F.layer_norm(
            expected_sum, (channels,), weight.float(), bias.float(), 1.0e-5
        ),
        atol=0.04,
        rtol=0.04,
    )


def test_post_norm_rejects_cpu_before_native_launch() -> None:
    x = torch.zeros(2, 8, dtype=torch.float16)
    affine = torch.zeros(8, dtype=torch.float16)
    with pytest.raises(ValueError, match="CUDA"):
        infer_post_norm_output_forward_varlen(x, x, affine, affine)
