# SPDX-License-Identifier: MIT
"""Public BF16 blocks: updates, packed inference, live graphs and independent VJP."""

import math

import pytest
import torch

import flashrwkv2 as fr
from flashrwkv2.post_norm.tilelang import norm

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
]


def parameters(channels, rank=32, vocab=257, *, device="cuda"):
    def value(*shape, scale=0.02):
        return (torch.randn(*shape, device=device) * scale).bfloat16().requires_grad_()

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
        result[f"{stage}_w"] = torch.ones(channels, device=device, dtype=torch.bfloat16, requires_grad=True)
        result[f"{stage}_b"] = value(channels)
    return result


def adapter_parameters(params, rank=4):
    """FP32 master A/B with nonzero B, so every adapter has a first-step VJP."""
    return {
        name: (
            (torch.randn(rank, params[name].shape[1], device=params[name].device) * 0.02).requires_grad_(),
            (torch.randn(params[name].shape[0], rank, device=params[name].device) * 0.02).requires_grad_(),
        )
        for name in ("r", "k", "v", "out", "cmix_k", "cmix_v", "head")
    }


def adapter_options(adapters, use_nora, mappings):
    if adapters is None:
        return {}
    result = {}
    for name, prefix in mappings:
        result.update({
            f"{prefix}lora_a": adapters[name][0],
            f"{prefix}lora_b": adapters[name][1],
            f"{prefix}lora_scale": 0.75,
            f"{prefix}use_nora": use_nora,
        })
    return result


_RKV_ADAPTERS = (("r", "receptance_"), ("k", "key_"), ("v", "value_"))
_CMIX_ADAPTERS = (("cmix_k", "key_"), ("cmix_v", "value_"))


def train_layer(x, res, params, initial, shifts, head_size, v_first, adapters=None, use_nora=False):
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
        **adapter_options(adapters, use_nora, _RKV_ADAPTERS),
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
        **adapter_options(adapters, use_nora, (("out", "output_"),)),
    )
    normalized, total = norm(
        readout, params["ln2_w"], params["ln2_b"], res=total, return_res=True,
    )
    cmix, final_cmix = fr.statetune_cmix_bf16(
        normalized, shifts[1], params["cmix_x"], params["cmix_k"], params["cmix_v"],
        **adapter_options(adapters, use_nora, _CMIX_ADAPTERS),
    )
    return cmix, total, final, mixed[-1], final_cmix, prepared[-1]


def train_block(x, res, params, initial, shifts, targets, head_size, v_first, adapters=None, use_nora=False):
    cmix, total, final, tmix_shift, cmix_shift, _ = train_layer(
        x, res, params, initial, shifts, head_size, v_first, adapters, use_nora,
    )
    hidden = norm(cmix, params["lnout_w"], params["lnout_b"], res=total)
    logits = fr.pretrain_head_linear_bf16(
        hidden, params["head"], **adapter_options(adapters, use_nora, (("head", ""),)),
    )
    logprobs = fr.head_logprobs(logits.view(-1, logits.shape[-1]), targets.reshape(-1))
    return logprobs.view_as(targets), logits, final, tmix_shift, cmix_shift


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
def infer_layer(
    x, res, params, state, shifts, cu, slots, head_size, v_first,
    adapters=None, use_nora=False, ticket=None,
):
    channels, heads = x.shape[-1], x.shape[-1] // head_size
    # Pack sliced batch rows before calling the contiguous varlen interfaces.
    x, res = x.contiguous(), res.contiguous()
    if v_first is not None:
        v_first = v_first.contiguous()
    if ticket is None:
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
        **adapter_options(adapters, use_nora, _RKV_ADAPTERS),
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
        **adapter_options(adapters, use_nora, (("out", "output_"),)),
    )
    total, cmix = fr.infer_cmix_forward_varlen(
        output, total, params["ln2_w"], params["ln2_b"], params["cmix_x"],
        params["cmix_k"], params["cmix_v"].t().contiguous(),
        shift_state_pool=shifts[1], cu_seqlens=cu, state_indices=slots, validated_metadata=ticket,
        **adapter_options(adapters, use_nora, _CMIX_ADAPTERS),
    )
    return cmix, total, prepared[-1]


@torch.no_grad()
def infer_block(
    x, res, params, state, shifts, cu, slots, targets, head_size, v_first,
    adapters=None, use_nora=False, ticket=None,
):
    cmix, total, _ = infer_layer(
        x, res, params, state, shifts, cu, slots, head_size, v_first, adapters, use_nora, ticket,
    )
    hidden = fr.infer_post_norm_output_forward_varlen(
        cmix, total, params["lnout_w"], params["lnout_b"],
    )
    logits = fr.infer_head_linear_all_forward_varlen(
        hidden, params["head"], **adapter_options(adapters, use_nora, (("head", ""),)),
    )
    return fr.head_logprobs(logits, targets), logits


@pytest.mark.parametrize("length", [1, 5])
def test_prepare_tuning_migration_preserves_block_state_logits_and_logprobs(monkeypatch, tmp_path, length):
    import hashlib
    import json
    from flashrwkv2._internal import tuning

    monkeypatch.setattr(tuning, "_profile_root", lambda: tmp_path / "new")
    monkeypatch.setattr(tuning, "_LOADED", {})
    torch.manual_seed(736)
    batch, channels, head_size = 2, 128, 64
    params = parameters(channels)
    x = torch.randn(batch * length, channels, device="cuda", dtype=torch.bfloat16) * .1
    res = torch.randn_like(x) * .1
    initial = torch.randn(batch, 2, 64, 64, device="cuda") * .01
    shifts = [torch.randn(batch, channels, device="cuda", dtype=torch.bfloat16) * .1 for _ in range(2)]
    slots = torch.arange(batch, device="cuda", dtype=torch.int32)
    cu = slots.new_tensor([0, length, batch * length])
    targets = torch.randint(257, (batch * length,), device="cuda")
    state, pools = state_pools(initial, shifts, slots, batch)
    snapshots = []

    @torch.no_grad()
    def call():
        for target, source in zip((state._components()[0], *pools), (initial, *shifts), strict=True):
            target.copy_(source)
        result = infer_block(x, res, params, state, pools, cu, slots, targets, head_size, None)
        snapshots.append(tuple(t.clone() for t in (*result, state._components()[0], *pools)))

    before = fr.prepare_tuning([call], mode="inspect")
    assert not before["complete"]
    canonical = {"linear": (16, 64, 2, 32), "tmix/wkv_prepare": (16, 64, 2, 32),
                 "tmix/wkv7": (64, 1), "tmix/tokenshift": 256,
                 "tmix/tokenshift/fused": 1, "post_norm": 1, "tmix/readout": 1}
    legacy = tmp_path / "legacy"
    for record in before["records"]:
        key = [record["key"][0][0], *record["key"][1:]]
        encoded = json.dumps(key, separators=(",", ":"))
        path = legacy / record["owner"] / (hashlib.sha256(encoded.encode()).hexdigest() + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"key": key, "config": canonical[record["owner"]], "bitwise_equal": True}))
    prepared = fr.prepare_tuning([call], mode="tune", legacy_dir=legacy)
    assert prepared["complete"]
    assert prepared["counts"]["revalidated"] == len(before["records"])
    tuning._LOADED.clear()
    loaded = fr.prepare_tuning([call])
    assert loaded["counts"]["hit"] == len(before["records"])
    assert all(value.dtype == torch.float32 for value in snapshots[0][:3])
    for actual in snapshots[1:]:
        for value, reference in zip(actual, snapshots[0], strict=True):
            dtype = torch.int32 if value.dtype == torch.float32 else torch.int16
            assert torch.equal(value.view(dtype), reference.view(dtype))


@pytest.mark.parametrize(
    "channels,head_size,vocab,length,use_vres",
    ((128, 64, 257, 17, False), (128, 64, 257, 17, True),
     (1024, 128, 257, 17, False), (4096, 64, 257, 17, True),
     (256, 256, 65536, 3, False)),
)
def test_block_bitwise_training_prefill_decode_and_batch(channels, head_size, vocab, length, use_vres):
    torch.manual_seed(731)
    batch = 2
    params = parameters(channels, vocab=vocab)
    x = (torch.randn(batch, length, channels, device="cuda") * 0.1).bfloat16().requires_grad_()
    res = torch.randn_like(x) * 0.1
    initial = torch.randn(batch, channels // head_size, head_size, head_size, device="cuda") * 0.01
    shifts = [torch.randn(batch, channels, device="cuda", dtype=torch.bfloat16) * 0.1 for _ in range(2)]
    targets = torch.randint(vocab, (batch, length), device="cuda")
    v_first = torch.randn_like(x) * 0.1 if use_vres else None
    expected = train_block(x, res, params, initial, shifts, targets, head_size, v_first)
    with torch.no_grad():
        recomputed = train_block(x, res, params, initial, shifts, targets, head_size, v_first)
    assert all(torch.equal(training, replay) for training, replay in zip(expected, recomputed))
    slots = torch.tensor([3, 1], device="cuda", dtype=torch.int32)

    split = min(5, length - 1)
    for chunks in ((length,), (split, length - split), (1,) * length):
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


@pytest.mark.parametrize("mode", ("base", "lora", "nora"))
def test_block_two_optimizer_updates_preserve_same_weight_alignment(mode):
    torch.manual_seed(732)
    channels, head_size, length = 128, 64, 5
    params = parameters(channels, rank=16)
    adapters = None if mode == "base" else adapter_parameters(params)
    use_nora = mode == "nora"
    if adapters is None:
        groups = {
            "tmix": [params[name] for name in ("r", "k", "v", "out", "mix")],
            "cmix": [params["cmix_k"], params["cmix_v"]],
            "head": [params["head"]],
        }
        optimized = list(params.values())
    else:
        for parameter in params.values():
            parameter.requires_grad_(False)
        groups = {name: list(pair) for name, pair in adapters.items()}
        optimized = [parameter for pair in adapters.values() for parameter in pair]
        assert all(parameter.dtype == torch.float32 for parameter in optimized)
    optimizer = torch.optim.SGD(optimized, lr=0.1)
    x = torch.randn(1, length, channels, device="cuda", dtype=torch.bfloat16) * 0.1
    res = torch.randn_like(x) * 0.1
    initial = torch.zeros(1, 2, 64, 64, device="cuda")
    shifts = [torch.zeros(1, channels, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    targets = torch.randint(257, (1, length), device="cuda")
    slots = torch.tensor([0], device="cuda", dtype=torch.int32)
    for step in range(3):
        trained = train_block(
            x, res, params, initial, shifts, targets, head_size, None, adapters, use_nora,
        )
        for chunks in ((length,), (1,) * length):
            state, pools = state_pools(initial, shifts, slots, 1)
            probabilities, logits = [], []
            start = 0
            for count in chunks:
                cu = torch.tensor([0, count], device="cuda", dtype=torch.int32)
                inferred = infer_block(
                    x[0, start:start + count], res[0, start:start + count],
                    params, state, pools, cu, slots, targets[0, start:start + count],
                    head_size, None, adapters, use_nora,
                )
                probabilities.append(inferred[0])
                logits.append(inferred[1])
                start += count
            assert torch.equal(trained[0][0], torch.cat(probabilities)), f"{mode}, weights {step}, chunks {chunks}"
            assert torch.equal(trained[1][0], torch.cat(logits)), f"{mode}, weights {step}, chunks {chunks}"
            for actual, expected in zip(
                (state._components()[0], *pools), trained[2:], strict=True,
            ):
                assert torch.equal(actual, expected)
        if step < 2:
            optimizer.zero_grad(set_to_none=True)
            before = {name: [p.detach().clone() for p in group] for name, group in groups.items()}
            (-trained[0].mean()).backward()
            for name, group in groups.items():
                for parameter in group:
                    assert parameter.grad is not None, f"{mode}/{name}: detached parameter"
                    assert torch.isfinite(parameter.grad).all(), f"{mode}/{name}: nonfinite VJP"
                    assert torch.count_nonzero(parameter.grad), f"{mode}/{name}: zero VJP"
            optimizer.step()
            for name, group in groups.items():
                changed = [
                    not torch.equal(old, parameter)
                    for old, parameter in zip(before[name], group, strict=True)
                ]
                assert (all(changed) if adapters is not None else any(changed)), f"{mode}/{name}: no update"


def test_two_layers_carry_real_v_first_through_prefill_decode_and_backward():
    torch.manual_seed(734)
    channels, head_size, batch, length = 128, 64, 2, 5
    params = [parameters(channels, rank=16) for _ in range(2)]
    x = (torch.randn(batch, length, channels, device="cuda") * 0.1).bfloat16().requires_grad_()
    res = torch.randn_like(x) * 0.1
    initial = [torch.randn(batch, 2, 64, 64, device="cuda") * 0.01 for _ in range(2)]
    shifts = [
        [torch.randn(batch, channels, device="cuda", dtype=torch.bfloat16) * 0.1 for _ in range(2)]
        for _ in range(2)
    ]
    targets = torch.randint(257, (batch, length), device="cuda")
    first = train_layer(x, res, params[0], initial[0], shifts[0], head_size, None)
    first[-1].retain_grad()
    expected = train_block(
        *first[:2], params[1], initial[1], shifts[1], targets, head_size, first[-1],
    )
    slots = torch.tensor([3, 1], device="cuda", dtype=torch.int32)
    for chunks in ((length,), (2, 3), (1,) * length):
        pooled = [state_pools(state, shift, slots, 5) for state, shift in zip(initial, shifts, strict=True)]
        probabilities, logits = [], []
        start = 0
        for count in chunks:
            cu = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * count
            first_infer = infer_layer(
                x[:, start:start + count].reshape(-1, channels),
                res[:, start:start + count].reshape(-1, channels),
                params[0], *pooled[0], cu, slots, head_size, None,
            )
            assert torch.equal(
                first[-1][:, start:start + count].reshape(-1, channels), first_infer[-1],
            )
            inferred = infer_block(
                *first_infer[:2], params[1], *pooled[1], cu, slots,
                targets[:, start:start + count].reshape(-1), head_size, first_infer[-1],
            )
            probabilities.append(inferred[0].view(batch, count))
            logits.append(inferred[1].view(batch, count, -1))
            start += count
        assert torch.equal(expected[0], torch.cat(probabilities, dim=1))
        assert torch.equal(expected[1], torch.cat(logits, dim=1))
        for (state, pools), finals in zip(pooled, (first[2:5], expected[2:]), strict=True):
            for actual, reference in zip((state._components()[0], *pools), finals, strict=True):
                assert torch.equal(reference, actual.index_select(0, slots.long()))
                assert torch.count_nonzero(actual[[0, 2, 4]]) == 0
    (-expected[0].mean()).backward()
    for name, parameter in (
        ("first value", first[-1]), ("first projection", params[0]["v"]),
        *((name, params[1][name]) for name in ("v0", "v1", "v2")),
    ):
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert torch.count_nonzero(parameter.grad), name


@pytest.mark.memcheck
def test_full_block_live_graph_keeps_all_three_state_pools_consistent():
    torch.manual_seed(735)
    channels, head_size, capacity, pool_size = 128, 64, 4, 2
    params = parameters(channels, rank=16)
    adapters = adapter_parameters(params)
    x = (torch.randn(capacity, channels, device="cuda") * 0.1).bfloat16()
    res = torch.randn_like(x) * 0.1
    targets = torch.randint(257, (capacity,), device="cuda")
    initial = torch.randn(pool_size, 2, 64, 64, device="cuda") * 0.01
    shifts = [torch.randn(pool_size, channels, device="cuda", dtype=torch.bfloat16) * 0.1 for _ in range(2)]
    physical_slots = torch.arange(pool_size, device="cuda", dtype=torch.int32)
    state, pools = state_pools(initial, shifts, physical_slots, pool_size)
    cu = torch.tensor([0, -1, -1, -1, -1], device="cuda", dtype=torch.int32)
    slots = torch.full((capacity,), 99, device="cuda", dtype=torch.int32)
    active_tokens = torch.zeros(1, device="cuda", dtype=torch.int32)
    active_sequences = torch.zeros_like(active_tokens)
    addresses = tuple(t.data_ptr() for t in (cu, slots, active_tokens, active_sequences))

    def execute():
        ticket = fr.prepare_tmix_wkv7_recurrent_metadata(
            cu, slots, state_pool_size=pool_size, token_capacity=capacity,
            sequence_capacity=capacity, max_seqlen_capacity=1,
            num_active_tokens=active_tokens, num_active_sequences=active_sequences,
        )
        result = infer_block(
            x, res, params, state, pools, cu, slots, targets, head_size, None,
            adapters, True, ticket,
        )
        return result, ticket

    # T=1 and equal token/sequence capacities exercise fused PostNorm+TokenShift
    # in both TMix and CMix. Warmup has zero active rows and invalid tail slots.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _, warmup_ticket = execute()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    assert warmup_ticket._is_graph() and warmup_ticket._max_seqlen() == 1
    for actual, original in zip((state._components()[0], *pools), (initial, *shifts), strict=True):
        assert torch.equal(actual, original)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured, ticket = execute()

    # The same captured addresses accept changing active counts, ignore invalid
    # inactive slots, and reject invalid/duplicate active slots without writes.
    for mapping in ((1, 0), (0,), (), (99,), (1, 1)):
        active = len(mapping)
        cu.copy_(torch.tensor([*range(active + 1), *([-1] * (capacity - active))], device="cuda", dtype=torch.int32))
        slots.copy_(torch.tensor([*mapping, *([99] * (capacity - active))], device="cuda", dtype=torch.int32))
        active_tokens.fill_(active)
        active_sequences.fill_(active)
        for actual, original in zip((state._components()[0], *pools), (initial, *shifts), strict=True):
            actual.copy_(original)
        graph.replay()
        torch.cuda.synchronize()
        valid = len(set(mapping)) == active and all(0 <= slot < pool_size for slot in mapping)
        if valid and active:
            eager_state, eager_pools = state_pools(initial, shifts, physical_slots, pool_size)
            eager = infer_block(
                x[:active], res[:active], params, eager_state, eager_pools,
                cu[:active + 1], slots[:active], targets[:active], head_size, None,
                adapters, True,
            )
            for actual, expected in zip(captured, eager, strict=True):
                assert torch.equal(actual[:active], expected)
            expected_pools = (eager_state._components()[0], *eager_pools)
            for actual, expected in zip((state._components()[0], *pools), expected_pools, strict=True):
                assert torch.equal(actual, expected)
            for slot in set(range(pool_size)) - set(mapping):
                for actual, original in zip((state._components()[0], *pools), (initial, *shifts), strict=True):
                    assert torch.equal(actual[slot], original[slot])
        else:
            for actual, original in zip((state._components()[0], *pools), (initial, *shifts), strict=True):
                assert torch.equal(actual, original)
        assert addresses == tuple(t.data_ptr() for t in (cu, slots, active_tokens, active_sequences))
        assert ticket._is_graph()


def torch_block_reference(
    x, res, params, initial, shifts, targets, head_size, v_first, adapters=None, use_nora=False,
):
    """Independent Torch equations, with explicit casts at the model's BF16 edges."""
    batch, length, channels = x.shape
    heads = channels // head_size

    def mm(value, weight, dtype=torch.bfloat16):
        return (value.float() @ weight.float().t()).to(dtype)

    def project(value, name, dtype=torch.bfloat16):
        base = mm(value, params[name], dtype)
        if adapters is None:
            return base
        a, b = adapters[name]
        if use_nora:
            a = a.float() / (torch.linalg.vector_norm(a.float(), dim=0, keepdim=True) + 1e-6)
        features = mm(value, a.bfloat16())
        update = mm(features, b.bfloat16(), torch.float32)
        return (base.float() + update * 0.75).to(dtype)

    def ln(value, stage, residual):
        total = (value.float() + residual.float()).bfloat16()
        centered = total.float() - total.float().mean(-1, keepdim=True)
        unit = centered * torch.rsqrt(centered.square().mean(-1, keepdim=True) + 1e-5)
        output = (unit * params[f"{stage}_w"].float() + params[f"{stage}_b"].float()).bfloat16()
        return output, total

    def shift(value, previous, mix):
        old = torch.cat((previous[:, None], value[:, :-1]), dim=1).float()
        return (value.float() + (old - value.float()) * mix.float()).bfloat16()

    normalized, total = ln(x, "ln1", res)
    mixed = [shift(normalized, shifts[0], mix) for mix in params["mix"].unbind()]
    r, key, value = [project(mixed[index], name) for index, name in ((0, "r"), (2, "k"), (3, "v"))]
    decay_hidden = torch.tanh(mm(mixed[1], params["w1"].t()).float()).bfloat16()
    decay = (mm(decay_hidden, params["w2"].t()).float() + params["w0"].float()).bfloat16()
    if v_first is not None:
        v_logits = mm(mm(mixed[3], params["v1"].t()), params["v2"].t()).float()
        v_gate = torch.sigmoid(v_logits + params["v0"].float())
        value = (value.float() + (v_first.float() - value.float()) * v_gate).bfloat16()
    rate_logits = mm(mm(mixed[4], params["a1"].t()), params["a2"].t())
    rate = torch.sigmoid(rate_logits.float() + params["a0"].float()).bfloat16()
    gate_hidden = torch.sigmoid(mm(mixed[5], params["g1"].t()).float()).bfloat16()
    gate = mm(gate_hidden, params["g2"].t())
    direction = (key.float() * params["k_k"].float()).view(batch, length, heads, head_size)
    direction = direction / torch.linalg.vector_norm(direction, dim=-1, keepdim=True).clamp_min(1e-12)
    recurrent_a = (-direction).bfloat16()
    recurrent_b = (direction * rate.float().view_as(direction)).bfloat16()
    key = (key.float() * (1.0 + (rate.float() - 1.0) * params["k_a"].float())).bfloat16()
    r_heads, k_heads, v_heads = [t.float().view(batch, length, heads, head_size) for t in (r, key, value)]
    retention = torch.exp(-math.exp(-0.5) * torch.sigmoid(decay.float())).view_as(r_heads)
    state = initial
    outputs = []
    for token in range(length):
        sa = (state * recurrent_a[:, token].float()[..., None]).sum(-2)
        state = (
            state * retention[:, token, :, :, None]
            + recurrent_b[:, token].float()[..., None] * sa[:, :, None, :]
            + k_heads[:, token, :, :, None] * v_heads[:, token, :, None, :]
        )
        outputs.append((state * r_heads[:, token, :, :, None]).sum(-2).bfloat16())
    recurrent = torch.stack(outputs, dim=1).float()
    centered = recurrent - recurrent.mean(-1, keepdim=True)
    unit = centered * torch.rsqrt(centered.square().mean(-1, keepdim=True) + 64e-5)
    rkv = (r_heads * k_heads * params["r_k"].float().view(heads, head_size)).sum(-1, keepdim=True)
    readout = (
        (unit.view_as(x) * params["lnx_w"].float() + params["lnx_b"].float()
         + (rkv * v_heads).view_as(x)) * gate.float()
    ).bfloat16()
    output = project(readout, "out")
    normalized_cmix, total = ln(output, "ln2", total)
    cmix_input = shift(normalized_cmix, shifts[1], params["cmix_x"])
    features = torch.relu(project(cmix_input, "cmix_k").float()).square().bfloat16()
    cmix = project(features, "cmix_v")
    hidden, _ = ln(cmix, "lnout", total)
    logits = project(hidden, "head", torch.float32)
    probabilities = logits.log_softmax(-1).gather(-1, targets[..., None]).squeeze(-1)
    return probabilities, logits, state, normalized[:, -1], normalized_cmix[:, -1]


@pytest.mark.parametrize("mode", ("base", "nora"))
def test_block_independent_torch_vjp_and_optimizer_step(mode):
    torch.manual_seed(736)
    channels, head_size, batch, length = 64, 64, 2, 3
    params = parameters(channels, rank=8, vocab=67)
    adapters = adapter_parameters(params) if mode == "nora" else None
    refs = {name: value.detach().clone().requires_grad_() for name, value in params.items()}
    ref_adapters = None if adapters is None else {
        name: tuple(value.detach().clone().requires_grad_() for value in pair)
        for name, pair in adapters.items()
    }
    x, res, first = [
        (torch.randn(batch, length, channels, device="cuda") * 0.1).bfloat16().requires_grad_()
        for _ in range(3)
    ]
    state = (torch.randn(batch, 1, 64, 64, device="cuda") * 0.01).requires_grad_()
    shifts = [
        (torch.randn(batch, channels, device="cuda") * 0.1).bfloat16().requires_grad_()
        for _ in range(2)
    ]
    inputs = (x, res, first, state, *shifts)
    reference_inputs = tuple(value.detach().clone().requires_grad_() for value in inputs)
    rx, rr, rf, rs, *rshifts = reference_inputs
    targets = torch.randint(67, (batch, length), device="cuda")
    actual = train_block(x, res, params, state, shifts, targets, head_size, first, adapters, mode == "nora")
    # Float32 Torch matmuls provide an independent real-valued implementation.
    # Disable TF32 so this check does not depend on backend precision settings.
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        expected = torch_block_reference(
            rx, rr, refs, rs, rshifts, targets, head_size, rf, ref_adapters, mode == "nora",
        )
        for observed, reference in zip(actual, expected, strict=True):
            torch.testing.assert_close(observed, reference, atol=3e-3, rtol=2e-2)
        # Final-state and shift cotangents exercise state VJPs in addition to the
        # language-model loss; both implementations receive identical signals.
        cotangents = (
            torch.randn_like(actual[0]), torch.randn_like(actual[1]) * 0.1,
            torch.randn_like(actual[2]) * 0.01,
            torch.randn_like(actual[3]) * 0.01, torch.randn_like(actual[4]) * 0.01,
        )
        actual_parameters = list(params.values())
        reference_parameters = list(refs.values())
        parameter_names = list(params)
        if adapters is not None:
            for name, pair in adapters.items():
                actual_parameters.extend(pair)
                reference_parameters.extend(ref_adapters[name])
                parameter_names.extend((f"{name}.A", f"{name}.B"))
        gradients = torch.autograd.grad(actual, (*inputs, *actual_parameters), cotangents)
        references = torch.autograd.grad(expected, (*reference_inputs, *reference_parameters), cotangents)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    for name, observed, reference in zip(
        ("x", "res", "v_first", "state", "tmix_shift", "cmix_shift", *parameter_names),
        gradients, references, strict=True,
    ):
        assert torch.isfinite(observed).all(), name
        reference_norm = torch.linalg.vector_norm(reference.float())
        assert reference_norm > 0, f"{name}: reference fixture has no VJP signal"
        relative_l2 = torch.linalg.vector_norm(observed.float() - reference.float()) / reference_norm
        # Separate BF16 cast/accumulation edges in the Torch graph contribute
        # rounding noise across the full block. Relative error also catches a
        # detached or zeroed branch regardless of its gradient's absolute size.
        assert relative_l2 < 5e-2, f"{name}: relative VJP error {relative_l2.item()}"
    before = [parameter.detach().clone() for parameter in actual_parameters]
    actual_optimizer = torch.optim.SGD(actual_parameters, lr=0.1)
    reference_optimizer = torch.optim.SGD(reference_parameters, lr=0.1)
    for parameter, gradient in zip(actual_parameters, gradients[len(inputs):], strict=True):
        parameter.grad = gradient
    for parameter, gradient in zip(reference_parameters, references[len(inputs):], strict=True):
        parameter.grad = gradient
    actual_optimizer.step()
    reference_optimizer.step()
    for name, observed, reference in zip(parameter_names, actual_parameters, reference_parameters, strict=True):
        torch.testing.assert_close(observed, reference, atol=5e-4, rtol=2e-2, msg=name)
    for names in (("r", "out"), ("cmix_k", "cmix_v"), ("head",)):
        assert any(
            not torch.equal(before[parameter_names.index(name)], params[name]) for name in names
        ), f"{names}: optimizer did not update this block stage"
