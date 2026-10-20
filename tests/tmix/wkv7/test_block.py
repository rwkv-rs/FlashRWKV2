# SPDX-License-Identifier: MIT
"""Public BF16 block path: training, packed prefill, decode and logprobs."""

import pytest
import torch

import flashrwkv2 as fr
from flashrwkv2.post_norm.tilelang import norm

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
]


def parameters(channels, rank=32, vocab=257):
    def value(*shape, scale=0.02):
        return (torch.randn(*shape, device="cuda") * scale).bfloat16().requires_grad_()

    result = {name: value(channels, channels) for name in ("r", "k", "v", "out")}
    result.update({name: value(channels, rank) for name in ("w1", "a1", "g1", "v1")})
    result.update({name: value(rank, channels) for name in ("w2", "a2", "g2", "v2")})
    result.update({
        name: value(channels, scale=0.1)
        for name in ("w0", "a0", "v0", "k_k", "k_a", "r_k", "cmix_x")
    })
    result["mix"] = value(6, channels, scale=0.1)
    result["cmix_k"] = value(4 * channels, channels)
    result["cmix_v"] = value(channels, 4 * channels)
    result["head"] = value(vocab, channels)
    for stage in ("ln1", "lnx", "ln2", "lnout"):
        result[f"{stage}_w"] = torch.ones(channels, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        result[f"{stage}_b"] = value(channels)
    return result


def train_block(x, res, params, initial, shifts, targets, head_size, v_first):
    batch, length, channels = x.shape
    heads = channels // head_size
    normalized, total = norm(
        x, params["ln1_w"], params["ln1_b"], res=res, return_res=True,
    )
    mixed = fr.statetune_tmix_tokenshift_bf16(normalized, shifts[0], *params["mix"].unbind())
    prepared = fr.pretrain_tmix_wkv_prepare_bf16(
        *mixed[:6], *(params[name] for name in ("r", "k", "v")),
        *(params[name] for name in ("w0", "w1", "w2", "a0", "a1", "a2", "g1", "g2", "k_k", "k_a")),
        v_first=v_first, v0=params["v0"], v1=params["v1"], v2=params["v2"], head_size=head_size,
    )
    cu = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * length
    recurrent, final = fr.rl_infctx_tmix_wkv7_recurrent_bf16(
        *(t.view(-1, heads, head_size) for t in prepared[:6]),
        state_pool=initial, cu_seqlens=cu,
    )
    r, _, k, v, _, _, gate, _ = prepared
    readout = fr.pretrain_tmix_readout_bf16(
        recurrent.view_as(x), r, k, v, params["r_k"].view(heads, head_size),
        params["lnx_w"], params["lnx_b"], gate,
        head_size=head_size, output_weight=params["out"],
    )
    normalized, total = norm(
        readout, params["ln2_w"], params["ln2_b"], res=total, return_res=True,
    )
    cmix, final_cmix = fr.statetune_cmix_bf16(
        normalized, shifts[1], params["cmix_x"], params["cmix_k"], params["cmix_v"],
    )
    hidden = norm(cmix, params["lnout_w"], params["lnout_b"], res=total)
    logits = fr.pretrain_head_linear_bf16(hidden, params["head"])
    logprobs = fr.head_logprobs(logits.view(-1, logits.shape[-1]), targets.reshape(-1))
    return logprobs.view(batch, length), logits, final, mixed[-1], final_cmix


def state_pools(initial, shifts, slots, pool_size):
    state = initial.new_zeros((pool_size, *initial.shape[1:]))
    state.index_copy_(0, slots.long(), initial.detach())
    shift_pools = []
    for shift in shifts:
        pool = shift.new_zeros(pool_size, shift.shape[-1])
        pool.index_copy_(0, slots.long(), shift.detach())
        shift_pools.append(pool)
    return fr.prepare_tmix_wkv7_recurrent_fp32io16_state_from_tensor(state), shift_pools


@torch.no_grad()
def infer_block(x, res, params, state, shifts, cu, slots, targets, head_size, v_first):
    channels, heads = x.shape[-1], x.shape[-1] // head_size
    ticket = fr.prepare_tmix_wkv7_recurrent_metadata(
        cu, slots, total_tokens=x.shape[0], state_pool_size=shifts[0].shape[0],
    )
    total, *mixed = fr.infer_tmix_postnorm_tokenshift_forward_varlen(
        x, res, params["ln1_w"], params["ln1_b"], *params["mix"].unbind(),
        shift_state_pool=shifts[0], cu_seqlens=cu, state_indices=slots, validated_metadata=ticket,
    )
    prepared = fr.infer_tmix_wkv_prepare_forward_varlen(
        *mixed, *(params[name] for name in ("r", "k", "v")),
        *(params[name].t() for name in ("w1", "a1", "g1", "v1", "w2", "a2", "g2", "v2")),
        *(params[name] for name in ("v0", "k_k", "a0", "k_a")),
        v_first=v_first, head_size=head_size,
    )
    recurrent = fr.infer_tmix_wkv7_recurrent_fp32io16_forward_varlen(
        *(t.view(-1, heads, head_size) for t in prepared[:6]),
        state=state, cu_seqlens=cu, state_indices=slots, validated_metadata=ticket,
        decay_bias=params["w0"].view(heads, head_size),
    )
    r, _, k, v, _, _, gate, _ = prepared
    output = fr.infer_tmix_readout_forward_varlen(
        recurrent.view(-1, channels), r, k, v, params["r_k"],
        params["lnx_w"], params["lnx_b"], gate, params["out"], head_size=head_size,
    )
    total, cmix = fr.infer_cmix_forward_varlen(
        output, total, params["ln2_w"], params["ln2_b"], params["cmix_x"],
        params["cmix_k"], params["cmix_v"].t().contiguous(),
        shift_state_pool=shifts[1], cu_seqlens=cu, state_indices=slots, validated_metadata=ticket,
    )
    hidden = fr.infer_post_norm_output_forward_varlen(
        cmix, total, params["lnout_w"], params["lnout_b"],
    )
    logits = fr.infer_head_linear_all_forward_varlen(hidden, params["head"])
    return fr.head_logprobs(logits, targets), logits


@pytest.mark.parametrize("channels,head_size", ((128, 64), (1024, 128), (4096, 64)))
@pytest.mark.parametrize("use_vres", (False, True))
def test_block_bitwise_training_prefill_decode_and_batch(channels, head_size, use_vres):
    torch.manual_seed(731)
    batch, length = 2, 17
    params = parameters(channels)
    x = (torch.randn(batch, length, channels, device="cuda") * 0.1).bfloat16().requires_grad_()
    res = torch.randn_like(x) * 0.1
    initial = torch.randn(batch, channels // head_size, head_size, head_size, device="cuda") * 0.01
    shifts = [torch.randn(batch, channels, device="cuda", dtype=torch.bfloat16) * 0.1 for _ in range(2)]
    targets = torch.randint(257, (batch, length), device="cuda")
    v_first = torch.randn_like(x) * 0.1 if use_vres else None
    expected = train_block(x, res, params, initial, shifts, targets, head_size, v_first)
    with torch.no_grad():
        recomputed = train_block(x, res, params, initial, shifts, targets, head_size, v_first)
    assert all(torch.equal(training, replay) for training, replay in zip(expected, recomputed))
    slots = torch.tensor([3, 1], device="cuda", dtype=torch.int32)

    for chunks in ((length,), (5, length - 5), (1,) * length):
        state, pools = state_pools(initial, shifts, slots, 5)
        probabilities, logits = [], []
        start = 0
        for count in chunks:
            stop = start + count
            cu = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * count
            probs, projected = infer_block(
                x[:, start:stop].reshape(-1, channels), res[:, start:stop].reshape(-1, channels),
                params, state, pools, cu, slots, targets[:, start:stop].reshape(-1), head_size,
                None if v_first is None else v_first[:, start:stop].reshape(-1, channels),
            )
            probabilities.append(probs.view(batch, count))
            logits.append(projected.view(batch, count, -1))
            start = stop
        assert torch.equal(expected[0], torch.cat(probabilities, dim=1)), f"logprobs: chunks={chunks}"
        assert torch.equal(expected[1], torch.cat(logits, dim=1)), f"logits: chunks={chunks}"
        assert torch.equal(expected[2], state._components()[0].index_select(0, slots.long()))
        assert torch.equal(expected[3], pools[0].index_select(0, slots.long()))
        assert torch.equal(expected[4], pools[1].index_select(0, slots.long()))
        assert torch.count_nonzero(state._components()[0][[0, 2, 4]]) == 0
        assert all(torch.count_nonzero(pool[[0, 2, 4]]) == 0 for pool in pools)

    state, pools = state_pools(initial[:1], [shift[:1] for shift in shifts], slots[:1], 5)
    alone, _ = infer_block(
        x[0], res[0], params, state, pools,
        torch.tensor([0, length], device="cuda", dtype=torch.int32),
        slots[:1], targets[0], head_size, None if v_first is None else v_first[0],
    )
    assert torch.equal(expected[0][0], alone)


def test_block_two_optimizer_updates_preserve_same_weight_alignment():
    torch.manual_seed(732)
    channels, head_size, length = 128, 64, 5
    params = parameters(channels, rank=16)
    optimizer = torch.optim.SGD(params.values(), lr=1e-2)
    x = torch.randn(1, length, channels, device="cuda", dtype=torch.bfloat16) * 0.1
    res = torch.randn_like(x) * 0.1
    initial = torch.zeros(1, 2, 64, 64, device="cuda")
    shifts = [torch.zeros(1, channels, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    targets = torch.randint(257, (1, length), device="cuda")
    cu = torch.tensor([0, length], device="cuda", dtype=torch.int32)
    slots = torch.tensor([0], device="cuda", dtype=torch.int32)
    for step in range(3):
        trained = train_block(x, res, params, initial, shifts, targets, head_size, None)[0]
        state, pools = state_pools(initial, shifts, slots, 1)
        inferred, _ = infer_block(
            x[0], res[0], params, state, pools, cu, slots, targets[0], head_size, None,
        )
        assert torch.equal(trained[0], inferred), f"weight version {step}"
        if step < 2:
            optimizer.zero_grad(set_to_none=True)
            before = params["head"].detach().clone()
            (-trained.mean()).backward()
            gradients = [parameter.grad for parameter in params.values() if parameter.grad is not None]
            assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
            assert any(torch.count_nonzero(gradient) for gradient in gradients)
            optimizer.step()
            assert not torch.equal(before, params["head"])
