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
    canonical = {key.replace("block.", "mtp.", 1): value for key, value in _weights()[0].items()}
    assert DraftLoop.from_state_dict(canonical, head, depth=1).state_dict().keys() == loop.state_dict().keys()
    canonical["block.norm_out.bias"] = canonical.pop("mtp.norm_out.bias")
    with pytest.raises(ValueError, match="mtp.norm_out.bias"):
        DraftLoop.from_state_dict(canonical, head, depth=1)


@torch.no_grad()
def _reference(weights, head, embed, hidden, token, depth, eps, adapter=None):
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
        normalized = layer_norm(hidden, "norm_out")
        logits = torch.mm(normalized, head.T, out_dtype=torch.float32)
        if adapter is not None:
            a, b, scale, use_nora = adapter
            if use_nora:
                a = a.float() / (a.float().norm(dim=0, keepdim=True) + 1e-6)
            update = torch.mm(F.linear(normalized, a.bfloat16()), b.bfloat16().T, out_dtype=torch.float32)
            logits.add_(update * scale)
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
        parent = torch.nn.Sequential(loop)
        replacement = parent.state_dict()
        replacement["0.lm_head"] = torch.zeros_like(head)
        parent.load_state_dict(replacement, assign=True)
        with pytest.raises(RuntimeError, match="capture"):
            loop.replay(hidden_a, token_a)
        loop.capture(embed)
        assert torch.equal(loop.replay(hidden_a, token_a), torch.zeros_like(expected_a))


@pytest.mark.cuda
@pytest.mark.cuda_graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_draft_loop_keeps_multiple_batch_graphs():
    weights, head = _weights()
    weights = {key: value.cuda().bfloat16() for key, value in weights.items()}
    head = head.cuda()
    table = torch.randn_like(head)

    def embed(ids):
        return F.embedding(ids, table)

    loop = DraftLoop.from_state_dict(weights, head, depth=4)
    batches = (1, 4, 64, 512)
    cases = {}
    outputs = {}
    for batch in batches:
        hidden = torch.randn(batch, head.shape[1], device="cuda", dtype=torch.bfloat16)
        token = torch.randint(head.shape[0], (batch,), device="cuda")
        expected = _reference(weights, head, embed, hidden, token, loop.depth, loop.eps)
        inputs = (hidden[0], token[0]) if batch == 1 else (hidden, token)
        cases[batch] = (*inputs, expected)
        outputs[batch] = loop(*inputs, embed)
        assert outputs[batch].shape == (batch, loop.depth)
        assert outputs[batch].dtype == torch.int64
        assert torch.equal(outputs[batch], expected)
        loop.capture(embed, batch_size=batch)

    for batch in reversed(batches):
        hidden, token, expected = cases[batch]
        replayed = loop.replay(hidden, token)
        assert replayed is outputs[batch]
        assert torch.equal(replayed, expected)
        if batch > 1:
            assert torch.equal(loop.replay(hidden.flip((0,)), token.flip((0,))), expected.flip((0,)))


@pytest.mark.cuda
@pytest.mark.cuda_graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("use_nora", (False, True))
def test_draft_loop_head_adapters(use_nora):
    weights, head = _weights()
    weights = {key: value.cuda().bfloat16() for key, value in weights.items()}
    head = head.cuda()
    table = torch.randn_like(head)

    def embed(ids):
        return F.embedding(ids, table)

    a = torch.randn(8, head.shape[1], device="cuda")
    b = torch.randn(head.shape[0], 8, device="cuda") * 0.1
    scale = 0.75
    loop = DraftLoop.from_state_dict(
        {key.replace("block.", "mtp.", 1): value for key, value in weights.items()},
        head, depth=4, lora_a=a, lora_b=b, lora_scale=scale, use_nora=use_nora,
    )
    hidden = torch.randn(4, head.shape[1], device="cuda", dtype=torch.bfloat16)
    token = torch.arange(4, device="cuda")
    expected = _reference(weights, head, embed, hidden, token, loop.depth, loop.eps, (a, b, scale, use_nora))
    assert not torch.equal(expected, _reference(weights, head, embed, hidden, token, loop.depth, loop.eps))
    assert torch.equal(loop(hidden, token, embed), expected)
    loop.capture(embed, batch_size=4)
    assert torch.equal(loop.replay(hidden, token), expected)


@pytest.mark.cuda
@pytest.mark.cuda_graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_draft_loop_peak_memory_uses_one_logits_plane():
    weights, head = _weights()
    head = head.cuda().repeat(512, 1)
    table = torch.randn_like(head[:128])

    def embed(ids):
        return F.embedding(ids, table)

    batch, depth = 512, 4
    loop = DraftLoop.from_state_dict(weights, head, depth=depth)
    hidden = torch.randn(batch, head.shape[1], device="cuda", dtype=torch.bfloat16)
    token = torch.randint(table.shape[0], (batch,), device="cuda")
    plane = batch * head.shape[0] * torch.float32.itemsize
    stacked = plane * depth
    loop(hidden, token, embed)
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    expected = loop(hidden, token, embed).clone()
    assert stacked == 512 * 1024**2
    assert plane <= torch.cuda.max_memory_allocated() - baseline < 2 * plane

    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    loop.capture(embed, batch_size=batch)
    assert plane <= torch.cuda.max_memory_allocated() - baseline < 2 * plane
    graph, _ = loop._captured[batch]
    reserved = sum(segment["total_size"] for segment in torch.cuda.memory_snapshot(mempool_id=graph.pool()))
    assert plane <= reserved < 2 * plane
    assert torch.equal(loop.replay(hidden, token), expected)
