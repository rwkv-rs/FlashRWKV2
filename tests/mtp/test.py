# SPDX-License-Identifier: MIT
"""LightMTP sidecar loading, eager drafting and graph replay."""

import pytest
import torch
import torch.nn.functional as F

from flashrwkv2.mtp import DraftLoop


def _weights():
    torch.manual_seed(682)
    channels, features, vocab = 64, 96, 128
    weights = {
        "block.proj.weight": torch.randn(channels, 2 * channels) * 0.08,
        "block.gate.weight": torch.randn(features, channels) * 0.08,
        "block.up.weight": torch.randn(features, channels) * 0.08,
        "block.down.weight": torch.randn(channels, features) * 0.08,
    }
    for name in ("norm_e", "norm_h", "norm_mlp", "norm_out"):
        weights[f"block.{name}.weight"] = 1 + torch.randn(channels) * 0.1
        weights[f"block.{name}.bias"] = torch.randn(channels) * 0.03
    head = torch.randn(vocab, channels).bfloat16() * 0.1
    return weights, head


def test_sidecar_weights_and_replay_lifecycle():
    weights, head = _weights()
    loop = DraftLoop.from_state_dict(weights, head, depth=4)
    assert loop.lm_head.data_ptr() == head.data_ptr()
    with pytest.raises(RuntimeError, match="capture"):
        loop.replay(head[:1], torch.zeros(1, dtype=torch.int64))
    with pytest.raises(ValueError, match="positive integer"):
        DraftLoop.from_state_dict(weights, head, depth=True)
    with pytest.raises(ValueError, match="block.norm_out.bias"):
        DraftLoop.from_state_dict(
            {key: value for key, value in weights.items() if key != "block.norm_out.bias"},
            head, depth=1,
        )
    weights["block.up.weight"] = weights["block.up.weight"][:, :-1]
    with pytest.raises(ValueError, match="block.up.weight"):
        DraftLoop.from_state_dict(weights, head, depth=1)


@torch.no_grad()
def _reference(weights, head, embed, hidden, token, depth, eps):
    def layer_norm(value, name):
        base = f"block.{name}"
        return F.layer_norm(
            value, (value.shape[-1],), weights[f"{base}.weight"], weights[f"{base}.bias"], eps,
        )

    drafts = []
    for _ in range(depth):
        joined = torch.cat((layer_norm(embed(token), "norm_e"), layer_norm(hidden, "norm_h")), dim=-1)
        residual = hidden + F.linear(joined, weights["block.proj.weight"])
        mlp = layer_norm(residual, "norm_mlp")
        gated = F.silu(F.linear(mlp, weights["block.gate.weight"])) * F.linear(mlp, weights["block.up.weight"])
        hidden = residual + F.linear(gated, weights["block.down.weight"])
        logits = torch.mm(layer_norm(hidden, "norm_out"), head.T, out_dtype=torch.float32)
        token = logits.argmax(-1)
        drafts.append(token)
    return torch.stack(drafts, dim=1)


@pytest.mark.cuda
@pytest.mark.cuda_graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("depth", [1, 4])
def test_draft_loop_matches_reference_and_uses_current_inputs(depth):
    weights, head = _weights()
    head = head.cuda()
    reference_weights = {key: value.cuda().bfloat16() for key, value in weights.items()}
    table = torch.randn_like(head)

    def embed(token):
        return F.embedding(token, table)

    eps = 1e-5
    loop = DraftLoop.from_state_dict(weights, head, depth=depth, eps=eps)
    assert loop.lm_head.data_ptr() == head.data_ptr()
    hidden_a, hidden_b = torch.randn(2, 1, head.shape[1], device="cuda", dtype=torch.bfloat16).unbind()
    token_a = torch.tensor([10], device="cuda", dtype=torch.int64)
    token_b = torch.tensor([29], device="cuda", dtype=torch.int64)
    expected_a = _reference(reference_weights, head, embed, hidden_a, token_a, depth, eps)
    expected_b = _reference(reference_weights, head, embed, hidden_b, token_b, depth, eps)
    assert not torch.equal(expected_a, expected_b)

    loop.capture(embed)
    first = loop.replay(hidden_a, token_a)
    assert torch.equal(first, expected_a)
    retained = first.clone()
    second = loop.replay(hidden_b, token_b)
    assert first is second
    assert torch.equal(second, expected_b)
    assert torch.equal(retained, expected_a)
    assert torch.equal(loop(hidden_a, token_a, embed), expected_a)
    assert torch.equal(loop(hidden_b, token_b, embed), expected_b)
    assert torch.equal(loop.replay(hidden_a, token_a), expected_a)
    if depth == 4:
        loop.cpu().cuda()
        with pytest.raises(RuntimeError, match="capture"):
            loop.replay(hidden_a, token_a)
        loop.capture(embed)
        assert torch.equal(loop.replay(hidden_b, token_b), expected_b)
