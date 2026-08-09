# FlashRWKV2 Kernel API

This document describes the public Python operator surface exported by
`flashrwkv2.__all__`. Every operator documented below is available from the
package root:

```python
from flashrwkv2 import <operator>
```

The owner shown for each entry identifies its implementation module; importing
from a submodule is not required. Native `_C` symbols, private helpers, internal
autograd classes, and names absent from `flashrwkv2.__all__` are not public API.
The current pretraining recurrent entry is `pretrain_recurrent_bf16`; the
removed `pretrain_recurrent_fp32io16` interface is not supported.

## Shared contracts

### Native extension and tensors

Accelerated calls require a built `flashrwkv2._C` extension. Unless an entry
says otherwise, tensors accepted by native wrappers must be CUDA, contiguous,
on one device, and have exactly the dtype and shape described by that entry.
Wrappers reject incompatible inputs instead of copying, casting, padding, or
making them contiguous implicitly.

Inference operators generally use FP16 token rows. The embedding and BF16 chunk
operators, and all `pretrain_*_bf16` operators, use BF16 where stated. Inference
wrappers do not define custom backward functions. The pretraining and StateTune
entries explicitly identified as autograd operators do.

### Packed sequences

Sequence-dependent inference operators use packed token rows. For a batch of
`B` non-empty sequences containing `N` total tokens:

| Value | Contract |
| --- | --- |
| token rows | leading shape `[N, ...]` |
| `cu_seqlens` | contiguous CUDA `int32`, shape `[B + 1]`, starting at `0` and ending at `N` |
| `state_indices` | contiguous CUDA `int32`, shape `[B]`, selecting one state slot per sequence |
| `max_seqlen` | optional positive upper bound for the longest packed sequence |

The native metadata validator also checks monotonic offsets, non-empty
sequences, slot bounds, duplicate slots, device identity, and the supplied token
and state-pool sizes where those properties apply.

### State pools and metadata tickets

Recurrent state pools have shape `[slots, H, D, D]`. Shift-state pools have
shape `[slots, C]`. Recurrent, chunk, TMix mix6, and stateful CMix inference
operators update the slots selected by `state_indices` in place. Callers own
allocation, slot reuse, synchronization, and request lifecycle.

`prepare_recurrent_metadata` returns an opaque native validation ticket. The
legacy static form snapshots fixed metadata. The capacity form can be prepared
on the consumer stream for a zero-active pre-capture warmup, then prepared
again with the same buffers during formal capture. It keeps `cu_seqlens`,
`state_indices`, and two CUDA `int32` active-count scalars live. Both forms are
bound to the same tensor identities, device, stream, state-pool size, and
launch capacity.
Graph validation covers only the active prefix; invalid counts, offsets,
lengths, slots, or duplicate slots fail closed before any stateful consumer
reads or writes recurrent state. Inactive capacity-tail rows never access a
state slot.

The FP16 recurrent operator reads a separate `int32` elapsed-state pool.
Advancing that pool is explicit through the two `advance_i32` APIs.

### Decay, mutation, and training boundaries

Inference recurrent, chunk, StateTune, and RL/Infctx interfaces named below
accept raw `decay_logits`; any retention transform is part of the operator.
`pretrain_recurrent_bf16` instead exposes the canonical clampw-v3 `w` input.

Inference state-pool metadata is not a training API. BF16 pretraining uses
dense `[B, T, C]` tensors and custom autograd. StateTune uses explicit chunk
metadata and clones its caller-owned initial state. RL/Infctx prepares its own
chunk metadata and returns a new final pool rather than mutating the input pool.

## Recurrent and chunk inference

### `infer_recurrent_fp32io16_forward_varlen`

- Import: `from flashrwkv2 import infer_recurrent_fp32io16_forward_varlen`
- Owner: `flashrwkv2.tmix.wkv7`
- Signature: `infer_recurrent_fp32io16_forward_varlen(r, decay_logits, k, v, a, b, *, state_pool, cu_seqlens, state_indices, scale=1.0, decay_bias=None, max_seqlen=None, validated_metadata=None) -> torch.Tensor`
- Contract: `r`, `decay_logits`, `k`, `v`, `a`, and `b` are matching packed `[N,H,D]` FP16 or BF16 tensors. `state_pool` is contiguous FP32 `[slots,H,D,D]`; supported `D` values are 64, 128, and 256. `decay_bias`, when supplied, is the per-head decay bias accepted by the native binding. Automatic dispatch uses the Albatross large and small-auto families. The upstream forced short-block family is retained as disabled `#if 0` reference code because this API exposes no corresponding selector.
- Result and mutation: returns output shaped like `v` and updates selected `state_pool` slots in place. No custom autograd.

### `infer_recurrent_fp16_forward_varlen`

- Import: `from flashrwkv2 import infer_recurrent_fp16_forward_varlen`
- Owner: `flashrwkv2.tmix.wkv7`
- Signature: `infer_recurrent_fp16_forward_varlen(r, decay_logits, k, v, a, b, *, state_pool, elapsed_state_pool, cu_seqlens, state_indices, scale=1.0, decay_bias=None, max_seqlen=None, validated_metadata=None) -> torch.Tensor`
- Contract: matching packed `[N,H,D]` token tensors, FP16 `state_pool [slots,H,D,D]`, and contiguous CUDA `int32 elapsed_state_pool [slots]`, with `D` in `{64,128,256}`. `D=64` preserves the Albatross clone/exact/seq-v2/one-cp/one-direct dispatch. `D=128/256` use the local 64-key-tiled FP16 recurrent family and never select an FP32-state fallback.
- Result and mutation: returns output shaped like `v` and updates selected state slots. Elapsed-state advancement remains explicit. No custom autograd.

### `infer_chunk_bf16_forward_varlen`

- Import: `from flashrwkv2 import infer_chunk_bf16_forward_varlen`
- Owner: `flashrwkv2.tmix.wkv7.chunk`
- Signature: `infer_chunk_bf16_forward_varlen(r, decay_logits, k, v, a, b, *, state_pool, cu_seqlens, state_indices, chunk_size=16, max_seqlen=None, scale=1.0, decay_bias=None, validated_metadata=None) -> torch.Tensor`
- Contract: all six token tensors are matching contiguous BF16 `[N,H,64]`; `state_pool` is BF16 `[slots,H,64,64]`. `chunk_size` is positive. Optional `decay_bias` is BF16 `[H,64]` or `[H*64]`.
- Result and mutation: returns `[N,H,64]` BF16 output and updates selected state slots in place. No custom autograd.

### `prepare_recurrent_metadata`

- Import: `from flashrwkv2 import prepare_recurrent_metadata`
- Owner: `flashrwkv2.tmix.wkv7`
- Signature: `prepare_recurrent_metadata(cu_seqlens, state_indices, *, state_pool_size, total_tokens=None, max_seqlen=None, token_capacity=None, sequence_capacity=None, max_seqlen_capacity=None, num_active_tokens=None, num_active_sequences=None) -> object`
- Contract: the static form requires `total_tokens` and validates one fixed packed launch. The capacity form requires all four capacity/count arguments, fixed metadata buffer shapes `[sequence_capacity+1]` and `[sequence_capacity]`, and one-element CUDA `int32` active-count tensors. It may run before capture for same-stream warmup or during formal capture.
- Result and mutation: returns one reusable native ticket. For warmup, set both active counts to zero before preparing and consuming the ticket; capacity may exceed the physical state-pool size because inactive metadata tails are ignored. Prepare the capacity form again with the same tensor addresses inside formal capture, where its validator reads live metadata and active counts on every replay. Callers update buffer contents in place rather than replacing tensors or moving state.

### `infer_recurrent_fp16_advance_i32`

- Import: `from flashrwkv2 import infer_recurrent_fp16_advance_i32`
- Owner: `flashrwkv2.tmix.wkv7`
- Signature: `infer_recurrent_fp16_advance_i32(elapsed_state, amount) -> None`
- Contract: `elapsed_state` is a non-empty contiguous CUDA `int32` tensor and `amount` is an integer.
- Result and mutation: increments every elapsed-state element in place and returns `None`.

### `infer_recurrent_fp16_advance_i32_varlen`

- Import: `from flashrwkv2 import infer_recurrent_fp16_advance_i32_varlen`
- Owner: `flashrwkv2.tmix.wkv7`
- Signature: `infer_recurrent_fp16_advance_i32_varlen(elapsed_state_pool, cu_seqlens, state_indices, *, total_tokens, validated_metadata=None) -> None`
- Contract: `elapsed_state_pool` is non-empty contiguous CUDA `int32 [slots]`; packed metadata selects the slots, and `total_tokens` is positive.
- Result and mutation: advances each selected slot by its packed sequence length and returns `None`.

### `infer_recurrent_add_vec_forward_varlen`

- Import: `from flashrwkv2 import infer_recurrent_add_vec_forward_varlen`
- Owner: `flashrwkv2.tmix.wkv7`
- Signature: `infer_recurrent_add_vec_forward_varlen(x, vec) -> torch.Tensor`
- Contract: FP16 `x [N,C]` with positive dimensions and even `C`; FP16 `vec [C]` on the same device.
- Result and mutation: returns `x + vec` with shape `[N,C]`; inputs are not mutated.

## Embedding and TimeMix preprocessing

### `infer_embedding_ln0_forward_varlen`

- Import: `from flashrwkv2 import infer_embedding_ln0_forward_varlen`
- Owner: `flashrwkv2.embedding`
- Signature: `infer_embedding_ln0_forward_varlen(embedding, weight, bias, *, eps=1e-5) -> torch.Tensor`
- Contract: BF16 `embedding [N,C]`, `weight [C]`, and `bias [C]`.
- Result and mutation: returns normalized packed rows with shape `[N,C]`; inputs are not mutated.

### `infer_tmix_mix6_forward_varlen`

- Import: `from flashrwkv2 import infer_tmix_mix6_forward_varlen`
- Owner: `flashrwkv2.tmix.mix6`
- Signature: `infer_tmix_mix6_forward_varlen(x, x_r, x_w, x_k, x_v, x_a, x_g, *, shift_state_pool, cu_seqlens, state_indices, max_seqlen=None, validated_metadata=None) -> tuple[torch.Tensor, ...]`
- Contract: FP16 `x [N,C]`, six FP16 coefficient vectors `[C]`, and FP16 shift state `[slots,C]` with packed metadata.
- Result and mutation: returns six `[N,C]` mixed tensors and updates selected shift-state slots to each sequence's last input row.

### `infer_tmix_mix6_add_layer_norm_forward_varlen`

- Import: `from flashrwkv2 import infer_tmix_mix6_add_layer_norm_forward_varlen`
- Owner: `flashrwkv2.tmix.mix6`
- Signature: `infer_tmix_mix6_add_layer_norm_forward_varlen(x, residual, weight, bias, x_r, x_w, x_k, x_v, x_a, x_g, *, shift_state_pool, cu_seqlens, state_indices, max_seqlen=None, eps=1e-5, validated_metadata=None) -> tuple[torch.Tensor, ...]`
- Contract: canonical fused FP16 path for `x` and `residual [1,4096]`, parameter vectors `[4096]`, one state index, and sequence length one. `eps` is finite and positive.
- Result and mutation: returns the summed row followed by six mixed `[1,4096]` tensors; updates the selected shift-state slot.

### `infer_tmix_kk_a_gate_forward_varlen`

- Import: `from flashrwkv2 import infer_tmix_kk_a_gate_forward_varlen`
- Owner: `flashrwkv2.tmix.kk_a_gate`
- Signature: `infer_tmix_kk_a_gate_forward_varlen(k, k_k, a0, a12, k_a, *, head_size=64, batch_size=1, max_seqlen=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]`
- Contract: FP16 `k` and `a12 [N,C]`, vectors `k_k`, `a0`, and `k_a [C]`; `head_size` is one of 64, 128, or 256 and divides `C`. Batch and maximum sequence lengths are positive. D64 retains the original launch, while D128/D256 use 2/4-warp head reductions.
- Result and mutation: returns the gated key, negative normalized key, and gated normalized key, each `[N,C]`; inputs are not mutated.

### `infer_tmix_lnx_rkvres_xg_forward_varlen`

- Import: `from flashrwkv2 import infer_tmix_lnx_rkvres_xg_forward_varlen`
- Owner: `flashrwkv2.tmix.lnx_rkvres_xg`
- Signature: `infer_tmix_lnx_rkvres_xg_forward_varlen(x, r, k, v, r_k, weight, bias, g, *, head_size=64, batch_size=1, max_seqlen=None) -> torch.Tensor`
- Contract: FP16 packed `x`, `r`, `k`, `v`, and `g [N,C]`; vectors `r_k`, `weight`, and `bias [C]`; `head_size` is one of 64, 128, or 256 and divides `C`. D64 preserves the Albatross dispatch; D128/D256 use 2/4-warp reductions for head normalization and the residual dot product.
- Result and mutation: returns fused head-wise normalization/residual/gate output `[N,C]`; inputs are not mutated.

### `infer_tmix_vres_gate_forward_varlen`

- Import: `from flashrwkv2 import infer_tmix_vres_gate_forward_varlen`
- Owner: `flashrwkv2.tmix.vres_gate`
- Signature: `infer_tmix_vres_gate_forward_varlen(v, v_first, v0, v12) -> torch.Tensor`
- Contract: FP16 `v`, `v_first`, and `v12 [N,C]`, plus FP16 `v0 [C]`.
- Result and mutation: returns the value-residual gated tensor `[N,C]`; inputs are not mutated.

### `infer_tmix_layer_norm_forward_varlen`

- Import: `from flashrwkv2 import infer_tmix_layer_norm_forward_varlen`
- Owner: `flashrwkv2.tmix.normalization`
- Signature: `infer_tmix_layer_norm_forward_varlen(x, weight, bias, *, eps=1e-5) -> torch.Tensor`
- Contract: FP16 `x [N,C]`, `weight [C]`, and `bias [C]`; `eps` is positive.
- Result and mutation: returns normalized `[N,C]`; inputs are not mutated.

### `infer_tmix_add_layer_norm_forward_varlen`

- Import: `from flashrwkv2 import infer_tmix_add_layer_norm_forward_varlen`
- Owner: `flashrwkv2.tmix.normalization`
- Signature: `infer_tmix_add_layer_norm_forward_varlen(x, residual, weight, bias, *, eps=1e-5, batch_size=None) -> tuple[torch.Tensor, torch.Tensor]`
- Contract: matching FP16 `x` and `residual [N,C]`, affine vectors `[C]`; optional `batch_size` is positive and no larger than `N`.
- Result and mutation: returns `(sum, normalized_sum)`, both `[N,C]`; inputs are not mutated.

### `infer_tmix_add_last_layer_norm_forward_varlen`

- Import: `from flashrwkv2 import infer_tmix_add_last_layer_norm_forward_varlen`
- Owner: `flashrwkv2.tmix.normalization`
- Signature: `infer_tmix_add_last_layer_norm_forward_varlen(x, residual, weight, bias, *, eps=1e-5) -> torch.Tensor`
- Contract: matching FP16 `x` and `residual [N,C]`, with FP16 `weight` and `bias [C]`.
- Result and mutation: returns the fused last-layer add/norm output `[N,C]`; inputs are not mutated.

### `infer_tmix_add_forward_varlen`

- Import: `from flashrwkv2 import infer_tmix_add_forward_varlen`
- Owner: `flashrwkv2.tmix.normalization`
- Signature: `infer_tmix_add_forward_varlen(x, residual) -> torch.Tensor`
- Contract: matching FP16 packed tensors `[N,C]`.
- Result and mutation: returns their elementwise sum; inputs are not mutated.

## Linear and low-rank inference

All entries in this section use contiguous CUDA FP16 tensors and return FP16
outputs. `M` denotes packed rows, `K` input features, `N` output features, and
`R` a low-rank dimension.

### `infer_tmix_linear_forward_varlen`

- Import: `from flashrwkv2 import infer_tmix_linear_forward_varlen`
- Owner: `flashrwkv2.tmix.linear`
- Signature: `infer_tmix_linear_forward_varlen(x, weight, *, weight_is_transposed=False) -> torch.Tensor`
- Contract: `x [M,K]`; weight is `[N,K]` by default or runtime-transposed `[K,N]` when requested.
- Result: returns `[M,N]`; inputs are not mutated.

### `infer_tmix_linear_attention_c2c_forward_varlen`

- Import: `from flashrwkv2 import infer_tmix_linear_attention_c2c_forward_varlen`
- Owner: `flashrwkv2.tmix.linear`
- Signature: `infer_tmix_linear_attention_c2c_forward_varlen(x, weight, *, lora_a=None, lora_b=None, lora_scale=1.0) -> torch.Tensor`
- Contract: attention-specific dispatch for `x [M,K]` and original-layout `weight [N,K]`. Optional vanilla-LoRA tensors must be supplied together in PEFT layout as `lora_a [R,K]` and `lora_b [N,R]`, with `1<=R<=512` and a finite float32-representable scale. Multiple active adapters, bias, dropout, DoRA, QLoRA, and per-sample adapter routing are outside this operator contract.
- Result: without LoRA, returns the unchanged base path `x @ weight.T`. With LoRA, returns `x @ weight.T + lora_scale * (x @ lora_a.T) @ lora_b.T` as contiguous FP16 `[M,N]`; inputs are immutable. Rank-out accumulates directly into the base output without a full `[M,N]` delta allocation. A validated zero scale reuses the base-only path.

### `infer_tmix_linear_ffn_key_forward_varlen`

- Import: `from flashrwkv2 import infer_tmix_linear_ffn_key_forward_varlen`
- Owner: `flashrwkv2.tmix.linear`
- Signature: `infer_tmix_linear_ffn_key_forward_varlen(x, weight) -> torch.Tensor`
- Contract and result: FFN-key dispatch for `x [M,K]` and `weight [N,K]`, returning `[M,N]` without mutation.

### `infer_tmix_linear_t_forward_varlen`

- Import: `from flashrwkv2 import infer_tmix_linear_t_forward_varlen`
- Owner: `flashrwkv2.tmix.linear`
- Signature: `infer_tmix_linear_t_forward_varlen(x, weight_t) -> torch.Tensor`
- Contract and result: `x [M,K]`, `weight_t [N,K]`, with `M <= 65535` because the custom kernel maps `M` to CUDA `grid.y`; returns `x @ weight_t.T` with shape `[M,N]`.

### `infer_tmix_linear_t_tanh_forward_varlen`

- Import: `from flashrwkv2 import infer_tmix_linear_t_tanh_forward_varlen`
- Owner: `flashrwkv2.tmix.linear`
- Signature: `infer_tmix_linear_t_tanh_forward_varlen(x, weight_t) -> torch.Tensor`
- Contract and result: applies tanh to `x [M,K]` before the `[N,K]` projection and returns `[M,N]`; `M <= 65535` because the custom kernel maps `M` to CUDA `grid.y`.

### `infer_tmix_linear_t_sigmoid_forward_varlen`

- Import: `from flashrwkv2 import infer_tmix_linear_t_sigmoid_forward_varlen`
- Owner: `flashrwkv2.tmix.linear`
- Signature: `infer_tmix_linear_t_sigmoid_forward_varlen(x, weight_t) -> torch.Tensor`
- Contract and result: applies sigmoid to `x [M,K]` before the `[N,K]` projection and returns `[M,N]`; `M <= 65535` because the custom kernel maps `M` to CUDA `grid.y`.

### `infer_tmix_linear_act_tanh_forward_varlen`

- Import: `from flashrwkv2 import infer_tmix_linear_act_tanh_forward_varlen`
- Owner: `flashrwkv2.tmix.linear`
- Signature: `infer_tmix_linear_act_tanh_forward_varlen(x) -> torch.Tensor`
- Contract and result: `x [M,K]` has an even element count; returns elementwise tanh with the same shape.

### `infer_tmix_linear_act_sigmoid_forward_varlen`

- Import: `from flashrwkv2 import infer_tmix_linear_act_sigmoid_forward_varlen`
- Owner: `flashrwkv2.tmix.linear`
- Signature: `infer_tmix_linear_act_sigmoid_forward_varlen(x) -> torch.Tensor`
- Contract and result: `x [M,K]` has an even element count; returns elementwise sigmoid with the same shape.

### `infer_tmix_linear_t_vres_forward_varlen`

- Import: `from flashrwkv2 import infer_tmix_linear_t_vres_forward_varlen`
- Owner: `flashrwkv2.tmix.linear`
- Signature: `infer_tmix_linear_t_vres_forward_varlen(x, weight_t, v, v_first, v0) -> torch.Tensor`
- Contract: `x [M,K]`, `weight_t [N,K]`, `v` and `v_first [M,N]`, and `v0 [N]`.
- Result: returns gated value-residual output `[M,N]`; inputs are not mutated.

### `infer_tmix_linear_rank_in_forward_varlen`

- Import: `from flashrwkv2 import infer_tmix_linear_rank_in_forward_varlen`
- Owner: `flashrwkv2.tmix.linear`
- Signature: `infer_tmix_linear_rank_in_forward_varlen(x, weight=None, weight_t=None) -> torch.Tensor`
- Contract: `x [M,K]`; at least one runtime `[K,R]` weight or original-layout `[R,K]` `weight_t` is required. If both are present they must describe the same projection.
- Result: returns rank features `[M,R]` using the shape-specific native dispatch.

### `infer_tmix_linear_rank_out_forward_varlen`

- Import: `from flashrwkv2 import infer_tmix_linear_rank_out_forward_varlen`
- Owner: `flashrwkv2.tmix.linear`
- Signature: `infer_tmix_linear_rank_out_forward_varlen(x, weight=None, weight_t=None) -> torch.Tensor`
- Contract: rank input `x [M,R]`; provide runtime `[R,N]` weight or original-layout `[N,R]` `weight_t`.
- Result: returns `[M,N]` using the shape-specific native dispatch.

### `infer_tmix_linear_rank_out_tanh_forward_varlen`

- Import: `from flashrwkv2 import infer_tmix_linear_rank_out_tanh_forward_varlen`
- Owner: `flashrwkv2.tmix.linear`
- Signature: `infer_tmix_linear_rank_out_tanh_forward_varlen(x, weight=None, weight_t=None) -> torch.Tensor`
- Contract and result: rank-out projection with tanh applied to `x`, returning `[M,N]` under the same weight rules as rank-out.

### `infer_tmix_linear_rank_out_sigmoid_forward_varlen`

- Import: `from flashrwkv2 import infer_tmix_linear_rank_out_sigmoid_forward_varlen`
- Owner: `flashrwkv2.tmix.linear`
- Signature: `infer_tmix_linear_rank_out_sigmoid_forward_varlen(x, weight=None, weight_t=None) -> torch.Tensor`
- Contract and result: rank-out projection with sigmoid applied to `x`, returning `[M,N]` under the same weight rules as rank-out.

### `infer_tmix_lowrank_in_forward_varlen`

- Import: `from flashrwkv2 import infer_tmix_lowrank_in_forward_varlen`
- Owner: `flashrwkv2.tmix.linear`
- Signature: `infer_tmix_lowrank_in_forward_varlen(x_w, x_a, x_g, w1, a1, g1, *, w1_runtime=None, a1_runtime=None, g1_runtime=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]`
- Contract: three packed sources `[M,C]`; each projection requires an original-layout `[R,C]` positional weight, a runtime-layout `[C,R]` keyword weight, or both. `R<=512`.
- Dispatch and result: `M<=7` uses the fused original-layout W/A/G family when all original weights are present. Other cases use the canonical Albatross large-row dispatcher and its available-layout policy. Returns three `[M,R]` tensors without mutating or converting inputs.

### `infer_tmix_lowrank_wagv_in_forward_varlen`

- Import: `from flashrwkv2 import infer_tmix_lowrank_wagv_in_forward_varlen`
- Owner: `flashrwkv2.tmix.linear`
- Signature: `infer_tmix_lowrank_wagv_in_forward_varlen(x_w, x_a, x_g, x_v, w1, a1, g1, v1, *, w1_runtime=None, a1_runtime=None, g1_runtime=None, v1_runtime=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]`
- Contract: four packed sources `[M,C]`; each projection accepts original `[R,C]`, runtime `[C,R]`, or both layouts. `R<=512`.
- Dispatch and result: the fused original-layout W/A/G/V family is selected only for `M<=7` with every original weight available. Larger or runtime-only inputs use canonical large-row dispatch and return four `[M,R]` tensors.

### `infer_tmix_lowrank_out_forward_varlen`

- Import: `from flashrwkv2 import infer_tmix_lowrank_out_forward_varlen`
- Owner: `flashrwkv2.tmix.linear`
- Signature: `infer_tmix_lowrank_out_forward_varlen(w1, a1, g1, w2, a2, g2, *, w2_runtime=None, a2_runtime=None, g2_runtime=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]`
- Contract: rank features `[M,R]`; each projection accepts original `[C,R]`, runtime `[R,C]`, or both layouts. `R<=512`.
- Dispatch and result: `M<=4` with all original weights uses fused rank-out. Otherwise native composition applies tanh to W and sigmoid to G before canonical large-row dispatch. Returns W/A/G `[M,C]` without timed layout conversion.

### `infer_tmix_lowrank_vres_forward_varlen`

- Import: `from flashrwkv2 import infer_tmix_lowrank_vres_forward_varlen`
- Owner: `flashrwkv2.tmix.linear`
- Signature: `infer_tmix_lowrank_vres_forward_varlen(w1, a1, g1, v1, w2, a2, g2, v2, v, v_first, v0, *, w2_runtime=None, a2_runtime=None, g2_runtime=None, v2_runtime=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]`
- Contract: W/A/G/V rank features `[M,R]`; each second-stage projection accepts original `[C,R]`, runtime `[R,C]`, or both layouts. Value tensors are `[M,C]`, `v0 [C]`, and `R<=512`.
- Dispatch and result: `M<=4` with every original weight uses fused W/A/G/V plus value residual. Other cases use canonical rank-out dispatch for W/A/G/V followed by `infer_tmix_vres_gate_forward_varlen`; no Torch/ATen fallback or layout conversion is performed.

For these four composite APIs, `M` is total packed rows, not a promise that a
single fused body accepts arbitrary `M`. If both layouts are supplied, native
dispatch chooses the fixed Albatross winner. Callers own layout preparation and
lifetime; runtime layouts should be inference-only non-persistent buffers and
must be rebuilt after weight, device, or dtype changes.

## ChannelMix inference

### `infer_cmix_mix_forward_varlen`

- Import: `from flashrwkv2 import infer_cmix_mix_forward_varlen`
- Owner: `flashrwkv2.cmix.mix`
- Signature: `infer_cmix_mix_forward_varlen(x, x_k, *, shift_state_pool, cu_seqlens, state_indices, max_seqlen=None, validated_metadata=None) -> torch.Tensor`
- Contract: FP16 `x [N,C]`, `x_k [C]`, and shift state `[slots,C]` with packed metadata.
- Result and mutation: returns mixed `[N,C]` and updates selected shift-state slots.

### `infer_cmix_add_layer_norm_mix_forward_varlen`

- Import: `from flashrwkv2 import infer_cmix_add_layer_norm_mix_forward_varlen`
- Owner: `flashrwkv2.cmix.mix`
- Signature: `infer_cmix_add_layer_norm_mix_forward_varlen(x, residual, weight, bias, x_k, *, shift_state_pool, cu_seqlens, state_indices, max_seqlen=None, eps=1e-5, validated_metadata=None) -> tuple[torch.Tensor, torch.Tensor]`
- Contract: canonical FP16 sequence-length-one path with `x` and `residual [B,4096]`, vectors `[4096]`, and one state slot per row. `eps` is finite and positive.
- Result and mutation: returns `(summed, mixed)` `[B,4096]` and updates selected shift-state slots.

### `infer_cmix_relu_square_forward_varlen`

- Import: `from flashrwkv2 import infer_cmix_relu_square_forward_varlen`
- Owner: `flashrwkv2.cmix.mix`
- Signature: `infer_cmix_relu_square_forward_varlen(x) -> torch.Tensor`
- Contract and result: FP16 `x [N,F]` with even element count; returns elementwise `relu(x)^2` without mutation.

### `infer_cmix_linear_ffn_down_forward_varlen`

- Import: `from flashrwkv2 import infer_cmix_linear_ffn_down_forward_varlen`
- Owner: `flashrwkv2.cmix.mix`
- Signature: `infer_cmix_linear_ffn_down_forward_varlen(x, weight) -> torch.Tensor`
- Contract and result: FP16 `x [N,K]`, runtime-layout `weight [K,C]`; returns `[N,C]` without mutation.

### `infer_cmix_sparse_forward_varlen`

- Import: `from flashrwkv2 import infer_cmix_sparse_forward_varlen`
- Owner: `flashrwkv2.cmix.sparse`
- Signature: `infer_cmix_sparse_forward_varlen(x, x_k, key_fc, value_fc, *, shift_state_pool, cu_seqlens, state_indices, max_seqlen=None, validated_metadata=None) -> torch.Tensor`
- Contract: FP16 `x [N,C]`, `x_k [C]`, `key_fc [F,C]`, `value_fc [F,C]`, and shift state `[slots,C]` with packed metadata. `N <= 65535` because active kernels map packed rows to CUDA `grid.y` and `grid.z`.
- Result and mutation: returns fused sparse CMix output `[N,C]` and updates selected shift-state slots.

### `infer_cmix_sparse_up_forward_varlen`

- Import: `from flashrwkv2 import infer_cmix_sparse_up_forward_varlen`
- Owner: `flashrwkv2.cmix.sparse`
- Signature: `infer_cmix_sparse_up_forward_varlen(x, x_k, key_fc, *, shift_state_pool, cu_seqlens, state_indices, max_seqlen=None, validated_metadata=None) -> torch.Tensor`
- Contract: FP16 `x [N,C]`, `x_k [C]`, `key_fc [F,C]`, and shift state `[slots,C]` with packed metadata. `N <= 65535` because the active up kernel maps packed rows to CUDA `grid.y`.
- Result and mutation: returns pre-activation `[N,F]` and updates selected shift-state slots.

### `infer_cmix_sparse_down_relu_forward_varlen`

- Import: `from flashrwkv2 import infer_cmix_sparse_down_relu_forward_varlen`
- Owner: `flashrwkv2.cmix.sparse`
- Signature: `infer_cmix_sparse_down_relu_forward_varlen(preact, value_fc, *, batch_size=None, max_seqlen=None) -> torch.Tensor`
- Contract: FP16 `preact [N,F]` and `value_fc [F,C]` with even `C`. `N <= 65535` because active down kernels map packed rows to CUDA `grid.y` or `grid.z`. `batch_size` and `max_seqlen` are either both omitted or both positive and large enough to cover `N` rows.
- Result: applies sparse ReLU-square/down projection and returns `[N,C]` without mutation.

## Head inference

### `infer_head_linear_forward_varlen`

- Import: `from flashrwkv2 import infer_head_linear_forward_varlen`
- Owner: `flashrwkv2.head.linear`
- Signature: `infer_head_linear_forward_varlen(x, weight) -> torch.Tensor`
- Contract and result: FP16 `x [N,C]` and `weight [vocab,C]`; returns logits `[N,vocab]`.

### `infer_head_linear_all_forward_varlen`

- Import: `from flashrwkv2 import infer_head_linear_all_forward_varlen`
- Owner: `flashrwkv2.head.linear`
- Signature: `infer_head_linear_all_forward_varlen(x, weight) -> torch.Tensor`
- Contract and result: canonical all-logits dispatch for FP16 `x [N,C]` and `weight [vocab,C]`, returning `[N,vocab]`.

### `infer_head_linear_last_forward_varlen`

- Import: `from flashrwkv2 import infer_head_linear_last_forward_varlen`
- Owner: `flashrwkv2.head.linear`
- Signature: `infer_head_linear_last_forward_varlen(x, weight, *, tokens_count) -> torch.Tensor`
- Contract: FP16 final rows `x [B,C]`, `weight [vocab,C]`, and positive integer `tokens_count` for caller dispatch.
- Result: returns final-row logits `[B,vocab]`; inputs are not mutated.

### `infer_head_last_norm_forward_varlen`

- Import: `from flashrwkv2 import infer_head_last_norm_forward_varlen`
- Owner: `flashrwkv2.head.linear`
- Signature: `infer_head_last_norm_forward_varlen(x, residual, last_indices, weight, bias, *, eps=1e-5) -> torch.Tensor`
- Contract: FP16 `x` and `residual [N,C]` with even `C`; CUDA `int64 last_indices [B]` contains absolute packed-row indices; affine vectors are FP16 `[C]`.
- Result: returns normalized selected rows `[B,C]`; inputs are not mutated. Index bounds are checked on device so the call is Graph-safe; an invalid index produces a NaN output row without an out-of-bounds read.

## Sampling inference

### `setup_sampling_states`

- Import: `from flashrwkv2 import setup_sampling_states`
- Owner: `flashrwkv2.sampling`
- Signature: `setup_sampling_states(seed, num_slots) -> torch.Tensor`
- Contract and result: initializes an explicit CUDA Philox state pool for positive `num_slots`. Callers retain the pool and assign requests through sampling `slot_indices`.

### `infer_sampling_temperature_topk_topp_forward_varlen`

- Import: `from flashrwkv2 import infer_sampling_temperature_topk_topp_forward_varlen`
- Owner: `flashrwkv2.sampling`
- Signature: `infer_sampling_temperature_topk_topp_forward_varlen(logits, states, slot_indices, *, temperature=1.0, top_k=-1, top_p=1.0, sample_capacity=None, num_active_samples=None) -> torch.Tensor`
- Contract: FP32 logits `[capacity,V]`, an explicit Philox state pool, and CUDA `int32 slot_indices [capacity]`; controls may be scalars or same-capacity CUDA vectors. `sample_capacity` and the one-element CUDA `int32 num_active_samples` must be supplied together.
- Result and mutation: returns CUDA `int32 [capacity]`. Active rows sample and advance their selected RNG slots. Inactive rows return `-1` without reading logits, controls, or RNG state. Invalid active counts, slots, or duplicate active slots make every row return `-1` without RNG mutation.

### `infer_sampling_six_parameter_forward_varlen`

- Import: `from flashrwkv2 import infer_sampling_six_parameter_forward_varlen`
- Owner: `flashrwkv2.sampling`
- Signature: `infer_sampling_six_parameter_forward_varlen(logits, penalties, states, slot_indices, *, presence_penalty=0.0, frequency_penalty=0.0, penalty_decay=0.996, temperature=1.0, top_k=-1, top_p=1.0, sample_capacity=None, num_active_samples=None) -> torch.Tensor`
- Contract: extends temperature/top-k/top-p sampling with FP32 `penalties [num_slots,V]`; each of the six controls may be a scalar or a same-capacity CUDA vector. `frequency_penalty` is Rapid-Sampling's additive repetition increment.
- Result and mutation: active rows advance their selected RNG and penalty slots. Inactive or invalid launches return the same `-1` sentinel contract and leave every inactive RNG and penalty slot unchanged.

## Pretraining

All entries in this section use custom autograd and return differentiable
outputs for their floating-point inputs unless an entry states otherwise.

### `pretrain_recurrent_bf16`

- Import: `from flashrwkv2 import pretrain_recurrent_bf16`
- Owner: `flashrwkv2.tmix.wkv7.pretrain`
- Signature: `pretrain_recurrent_bf16(r, w, k, v, a, b, *, head_size=64) -> torch.Tensor`
- Contract: six matching contiguous CUDA BF16 tensors `[B,T,C]`; `head_size` is one of 64, 128, or 256 and divides `C`; `T` is divisible by the canonical chunk length 16. `w` is the clampw-v3 input, not raw `decay_logits`. D64 retains clampw-v3, D128 follows RWKV-LM `rwkv7_clampw128_v2`, and D256 is a local warp-tiled generalization of the same recurrence and storage contract. Its backward materializes only the FP32 `dSb [B,T,H,D]` intermediate and runs independent value-column and key-row scans; no FP32-state fallback is used.
- Result and autograd: returns BF16 `[B,T,C]` and supplies gradients for all six inputs. No caller-owned state is mutated.

### `pretrain_tmix_a_gate_bf16`

- Import: `from flashrwkv2 import pretrain_tmix_a_gate_bf16`
- Owner: `flashrwkv2.tmix.a_gate`
- Signature: `pretrain_tmix_a_gate_bf16(a0, a12) -> torch.Tensor`
- Contract and result: BF16 `a0 [C]` and `a12 [B,T,C]`; returns the sigmoid gate `[B,T,C]` with gradients for both inputs.

### `pretrain_tmix_vres_gate_bf16`

- Import: `from flashrwkv2 import pretrain_tmix_vres_gate_bf16`
- Owner: `flashrwkv2.tmix.vres_gate`
- Signature: `pretrain_tmix_vres_gate_bf16(value, first_value, v0, v12)`
- Contract: BF16 `value`, `first_value`, and `v12 [B,T,C]`, plus `v0 [C]`.
- Result and autograd: returns the value-residual gate output `[B,T,C]` with gradients for all four inputs.

### `pretrain_tmix_mix6_bf16`

- Import: `from flashrwkv2 import pretrain_tmix_mix6_bf16`
- Owner: `flashrwkv2.tmix.mix6`
- Signature: `pretrain_tmix_mix6_bf16(x, x_r, x_w, x_k, x_v, x_a, x_g) -> tuple[torch.Tensor, ...]`
- Contract: BF16 `x [B,T,C]` and six BF16 vectors `[C]`.
- Result and autograd: returns six mixed `[B,T,C]` tensors and supplies gradients for all inputs.

### `pretrain_tmix_kk_pre_bf16`

- Import: `from flashrwkv2 import pretrain_tmix_kk_pre_bf16`
- Owner: `flashrwkv2.tmix.kk_pre`
- Signature: `pretrain_tmix_kk_pre_bf16(key, key_scale, learning_rate, learning_rate_scale, *, head_size=64)`
- Contract: BF16 `key` and `learning_rate [B,T,C]`; BF16 scale vectors `[C]`; `head_size` is one of 64, 128, or 256 and divides `C`.
- Result and autograd: returns `(new_key, negative_direction, scaled_direction)`, each `[B,T,C]`, with gradients for all inputs.

### `pretrain_tmix_lnx_rkvres_xg_bf16`

- Import: `from flashrwkv2 import pretrain_tmix_lnx_rkvres_xg_bf16`
- Owner: `flashrwkv2.tmix.lnx_rkvres_xg.pretrain`
- Signature: `pretrain_tmix_lnx_rkvres_xg_bf16(x, r, k, v, residual_scale, weight, bias, g, *, head_size=64)`
- Contract: matching BF16 `x`, `r`, `k`, `v`, and `g [B,T,C]`; `head_size` is one of 64, 128, or 256 and divides `C`; `residual_scale [C/head_size,head_size]`; affine vectors `[C]`.
- Result and autograd: returns fused `[B,T,C]` output with gradients for requested floating inputs.

### `pretrain_cmix_bf16`

- Import: `from flashrwkv2 import pretrain_cmix_bf16`
- Owner: `flashrwkv2.cmix.mix`
- Signature: `pretrain_cmix_bf16(x, x_k, key_weight, value_weight) -> torch.Tensor`
- Contract: BF16 `x [B,T,C]`, `x_k [C]`, `key_weight [4C,C]`, and `value_weight [C,4C]`.
- Result and autograd: returns BF16 CMix output `[B,T,C]` and supplies gradients for all four inputs.

### `pretrain_head_l2wrap_ce_bf16`

- Import: `from flashrwkv2 import pretrain_head_l2wrap_ce_bf16`
- Owner: `flashrwkv2.head.l2wrap_ce`
- Signature: `pretrain_head_l2wrap_ce_bf16(hidden, weight, targets, *, chunk_rows=4096)`
- Contract: BF16 `hidden [B,T,C]`, BF16 `weight [65536,C]`, CUDA `int64 targets` with one target per hidden row, and positive `chunk_rows`.
- Result and autograd: returns scalar CE/L2Wrap loss with gradients for `hidden` and `weight`; targets are non-differentiable.

### `pretrain_l2wrap_ce_bf16`

- Import: `from flashrwkv2 import pretrain_l2wrap_ce_bf16`
- Owner: `flashrwkv2.loss.l2wrap_ce`
- Signature: `pretrain_l2wrap_ce_bf16(logits, targets) -> torch.Tensor`
- Contract: BF16 or FP32 `logits [...,vocab]` and CUDA `int64 targets` with one valid target per logits row.
- Result and autograd: returns scalar CE/L2Wrap loss with gradients for logits; targets are non-differentiable.

## StateTune

### `statetune_tmix_mix6_bf16`

- Import: `from flashrwkv2 import statetune_tmix_mix6_bf16`
- Owner: `flashrwkv2.tmix.mix6`
- Signature: `statetune_tmix_mix6_bf16(x, initial_shift, x_r, x_w, x_k, x_v, x_a, x_g) -> tuple[torch.Tensor, ...]`
- Contract: contiguous CUDA BF16 `x [B,T,C]`, BF16 `initial_shift [B,C]`, and six BF16 coefficient vectors `[C]`; `B,T,C` are positive and `C` is even.
- Result and autograd: returns six mixed tensors `[B,T,C]` plus `next_shift = x[:,-1] [B,C]`. Gradients cover `x`, `initial_shift`, and all coefficient vectors, including the `next_shift` contribution to the last token. Inputs are not mutated.

### `statetune_cmix_bf16`

- Import: `from flashrwkv2 import statetune_cmix_bf16`
- Owner: `flashrwkv2.cmix.mix`
- Signature: `statetune_cmix_bf16(x, initial_shift, x_k, key_weight, value_weight) -> tuple[torch.Tensor, torch.Tensor]`
- Contract: contiguous CUDA BF16 `x [B,T,C]`, `initial_shift [B,C]`, `x_k [C]`, `key_weight [4C,C]`, and `value_weight [C,4C]`; `B,T,C` are positive and `C` is even.
- Result and autograd: returns the complete BF16 ChannelMix output `[B,T,C]` and `next_shift = x[:,-1] [B,C]`. Gradients cover all five inputs, including both state boundaries. Inputs are not mutated.

### `statetune_recurrent_fp32io16`

- Import: `from flashrwkv2 import statetune_recurrent_fp32io16`
- Owner: `flashrwkv2.tmix.wkv7.statetune`
- Signature: `statetune_recurrent_fp32io16(initial_state, sequence_chunk_offsets, chunk_token_starts, chunk_token_ends, r, decay_logits, k, v, a, b, *, scale=1.0)`
- Contract: FP32 `initial_state [B,H,D,D]` with `D` in `{64,128,256}`; matching FP16 or BF16 tokens `[N,H,D]`; CUDA `int32 sequence_chunk_offsets [B+1]` and matching chunk start/end arrays. `scale` is finite.
- Result and autograd: returns `(output, final_state, boundary, state_dot_a)`. Output matches token shape, final state matches `initial_state`, boundary is `[chunks,H,D,D]`, and `state_dot_a` is `[N,H,D]`. Boundary and `state_dot_a` are non-differentiable; gradients include the initial state and six token inputs. The input initial state is not mutated.

## RL/Infctx

### `rl_infctx_chunk_fp32io16`

- Import: `from flashrwkv2 import rl_infctx_chunk_fp32io16`
- Owner: `flashrwkv2.rl_infctx.wkv7`
- Signature: `rl_infctx_chunk_fp32io16(r, decay_logits, k, v, a, b, *, state_pool=None, cu_seqlens, state_indices=None, chunk_size=16, strategy='recompute', scale=1.0, decay_bias=None)`
- Contract: matching FP16 or BF16 packed tensors `[N,H,D]`, with `D` in `{64,128,256}`; packed offsets; `chunk_size` in `{16,32,64}`; strategy `materialized` or `recompute`; finite scale. Optional decay bias matches token dtype and is `[H,D]` or `[H*D]`. Missing state indices default to `0..B-1`; missing state pool allocates zero FP32 state. D64 retains the existing chunk selection. D128/D256 use 64-wide state tiles for boundary/state propagation and backward replay; no D256-by-D256 shared allocation is used.
- Result and mutation: returns `(output, final_pool)`. The output matches token shape; `final_pool` is a cloned FP32 pool with selected slots replaced. The input `state_pool` is not mutated. This forward-only wrapper does not define custom autograd.

### `rl_infctx_chunk_fp32io16_factor_recompute`

- Import: `from flashrwkv2 import rl_infctx_chunk_fp32io16_factor_recompute`
- Owner: `flashrwkv2.rl_infctx.wkv7`
- Signature: `rl_infctx_chunk_fp32io16_factor_recompute(*args, **kwargs)`
- Contract: accepts the same arguments as `rl_infctx_chunk_fp32io16` and forces `strategy='recompute'`, overriding a supplied strategy keyword.
- Result and mutation: returns the same `(output, final_pool)` pair and preserves the input state pool.

## Detailed parameter tables

The tables below are the parameter-level reference for every public callable.
Unless stated otherwise, tensor arguments follow the CUDA, contiguity, and
same-device rules in [Shared contracts](#shared-contracts). `None` in the dtype
column means that the argument is optional.

### Recurrent and chunk parameters

#### `infer_recurrent_fp32io16_forward_varlen`

| Parameter | Dtype | Shape | Description |
| --- | --- | --- | --- |
| `r` | fp16/bf16 | `[N,H,D]` | Receptance rows. |
| `decay_logits` | fp16/bf16 | `[N,H,D]` | Raw decay logits. |
| `k` | fp16/bf16 | `[N,H,D]` | Key rows. |
| `v` | fp16/bf16 | `[N,H,D]` | Value rows and output shape template. |
| `a` | fp16/bf16 | `[N,H,D]` | Low-rank state-update factor. |
| `b` | fp16/bf16 | `[N,H,D]` | Low-rank state-update factor. |
| `state_pool` | fp32 | `[slots,H,D,D]` | Recurrent state, updated in place. |
| `cu_seqlens` | int32 | `[B+1]` | Packed sequence offsets. |
| `state_indices` | int32 | `[B]` | State slot selected by each sequence. |
| `scale` | float | scalar | Output scale; default `1.0`. |
| `decay_bias` | fp16/bf16/None | `[H,D]` or `[H*D]` | Optional per-head decay-logit bias. |
| `max_seqlen` | int/None | scalar | Optional positive maximum sequence length. |
| `validated_metadata` | object/None | opaque | Optional reusable metadata ticket. |

#### `infer_recurrent_fp16_forward_varlen`

| Parameter | Dtype | Shape | Description |
| --- | --- | --- | --- |
| `r` | fp16/bf16 | `[N,H,64]` | Receptance rows. |
| `decay_logits` | fp16/bf16 | `[N,H,64]` | Raw decay logits. |
| `k` | fp16/bf16 | `[N,H,64]` | Key rows. |
| `v` | fp16/bf16 | `[N,H,64]` | Value rows and output shape template. |
| `a` | fp16/bf16 | `[N,H,64]` | Low-rank state-update factor. |
| `b` | fp16/bf16 | `[N,H,64]` | Low-rank state-update factor. |
| `state_pool` | fp16 | `[slots,H,64,64]` | Recurrent state, updated in place. |
| `elapsed_state_pool` | int32 | `[slots]` | Per-slot dither phase read by the kernel. |
| `cu_seqlens` | int32 | `[B+1]` | Packed sequence offsets. |
| `state_indices` | int32 | `[B]` | State slot selected by each sequence. |
| `scale` | float | scalar | Output scale; default `1.0`. |
| `decay_bias` | fp16/bf16/None | `[H,64]` or `[H*64]` | Optional decay-logit bias. |
| `max_seqlen` | int/None | scalar | Optional positive maximum sequence length. |
| `validated_metadata` | object/None | opaque | Optional reusable metadata ticket. |

#### `infer_chunk_bf16_forward_varlen`

| Parameter | Dtype | Shape | Description |
| --- | --- | --- | --- |
| `r` | bf16 | `[N,H,64]` | Receptance rows. |
| `decay_logits` | bf16 | `[N,H,64]` | Raw decay logits. |
| `k` | bf16 | `[N,H,64]` | Key rows. |
| `v` | bf16 | `[N,H,64]` | Value rows and output shape template. |
| `a` | bf16 | `[N,H,64]` | Low-rank state-update factor. |
| `b` | bf16 | `[N,H,64]` | Low-rank state-update factor. |
| `state_pool` | bf16 | `[slots,H,64,64]` | Chunk state, updated in place. |
| `cu_seqlens` | int32 | `[B+1]` | Packed sequence offsets. |
| `state_indices` | int32 | `[B]` | State slot selected by each sequence. |
| `chunk_size` | int | scalar | Positive chunk length; default `16`. |
| `max_seqlen` | int/None | scalar | Optional positive maximum sequence length. |
| `scale` | float | scalar | Finite output scale; default `1.0`. |
| `decay_bias` | bf16/None | `[H,64]` or `[H*64]` | Optional decay-logit bias. |
| `validated_metadata` | object/None | opaque | Optional static snapshot or live Graph metadata ticket. |

#### `prepare_recurrent_metadata`

| Parameter | Dtype | Shape | Description |
| --- | --- | --- | --- |
| `cu_seqlens` | int32 | `[B+1]` | Packed sequence offsets to validate. |
| `state_indices` | int32 | `[B]` | State slots to validate. |
| `total_tokens` | int | scalar | Expected positive packed row count. |
| `state_pool_size` | int | scalar | Number of available state slots. |
| `max_seqlen` | int/None | scalar | Optional expected maximum sequence length. |
| `token_capacity` | int/None | scalar | Fixed Graph token-row capacity. |
| `sequence_capacity` | int/None | scalar | Fixed Graph sequence capacity. |
| `max_seqlen_capacity` | int/None | scalar | Positive per-sequence Graph length bound. |
| `num_active_tokens` | int32/None | one element | Live CUDA Graph active-token count. |
| `num_active_sequences` | int32/None | one element | Live CUDA Graph active-sequence count. |

#### `infer_recurrent_fp16_advance_i32`

| Parameter | Dtype | Shape | Description |
| --- | --- | --- | --- |
| `elapsed_state` | int32 | any non-empty | Elapsed counters updated in place. |
| `amount` | int | scalar | Increment applied to every counter. |

#### `infer_recurrent_fp16_advance_i32_varlen`

| Parameter | Dtype | Shape | Description |
| --- | --- | --- | --- |
| `elapsed_state_pool` | int32 | `[slots]` | Counters updated in place. |
| `cu_seqlens` | int32 | `[B+1]` | Offsets whose differences are the increments. |
| `state_indices` | int32 | `[B]` | Counter slot selected by each sequence. |
| `total_tokens` | int | scalar | Expected positive packed row count. |
| `validated_metadata` | object/None | opaque | Optional reusable metadata ticket. |

#### `infer_recurrent_add_vec_forward_varlen`

| Parameter | Dtype | Shape | Description |
| --- | --- | --- | --- |
| `x` | fp16 | `[N,C]` | Packed input rows; `C` is even. |
| `vec` | fp16 | `[C]` | Vector broadcast across rows. |

### Embedding and TimeMix preprocessing parameters

#### `infer_embedding_ln0_forward_varlen`

| Parameter | Dtype | Shape | Description |
| --- | --- | --- | --- |
| `embedding` | bf16 | `[N,C]` | Packed embedding rows. |
| `weight` | bf16 | `[C]` | Layer-normalization weight. |
| `bias` | bf16 | `[C]` | Layer-normalization bias. |
| `eps` | float | scalar | Numerical epsilon; default `1e-5`. |

#### `infer_tmix_mix6_forward_varlen`

| Parameter | Dtype | Shape | Description |
| --- | --- | --- | --- |
| `x` | fp16 | `[N,C]` | Packed token rows. |
| `x_r`, `x_w`, `x_k`, `x_v`, `x_a`, `x_g` | fp16 | `[C]` each | Six TimeMix coefficient vectors. |
| `shift_state_pool` | fp16 | `[slots,C]` | Previous-token state, updated in place. |
| `cu_seqlens` | int32 | `[B+1]` | Packed sequence offsets. |
| `state_indices` | int32 | `[B]` | Shift-state slots. |
| `max_seqlen` | int/None | scalar | Optional maximum sequence length. |
| `validated_metadata` | object/None | opaque | Optional reusable metadata ticket. |

#### `infer_tmix_mix6_add_layer_norm_forward_varlen`

| Parameter | Dtype | Shape | Description |
| --- | --- | --- | --- |
| `x` | fp16 | `[1,4096]` | Current token row. |
| `residual` | fp16 | `[1,4096]` | Residual row added before normalization. |
| `weight`, `bias` | fp16 | `[4096]` each | Layer-normalization affine parameters. |
| `x_r`, `x_w`, `x_k`, `x_v`, `x_a`, `x_g` | fp16 | `[4096]` each | Six TimeMix coefficient vectors. |
| `shift_state_pool` | fp16 | `[slots,4096]` | Previous-token state, updated in place. |
| `cu_seqlens` | int32 | `[2]` | Single-sequence offsets for one token. |
| `state_indices` | int32 | `[1]` | Selected shift-state slot. |
| `max_seqlen` | int/None | scalar | Must resolve to `1`. |
| `eps` | float | scalar | Positive finite epsilon; default `1e-5`. |
| `validated_metadata` | object/None | opaque | Optional reusable metadata ticket. |

#### `infer_tmix_kk_a_gate_forward_varlen`

| Parameter | Dtype | Shape | Description |
| --- | --- | --- | --- |
| `k` | fp16 | `[N,C]` | Packed keys; `C` is divisible by 64. |
| `k_k` | fp16 | `[C]` | Key-normalization scale. |
| `a0` | fp16 | `[C]` | Gate bias. |
| `a12` | fp16 | `[N,C]` | Per-token gate term. |
| `k_a` | fp16 | `[C]` | Key interpolation scale. |
| `batch_size` | int | scalar | Positive logical batch size; default `1`. |
| `max_seqlen` | int/None | scalar | Positive dispatch extent; defaults to `N`. |

#### `infer_tmix_lnx_rkvres_xg_forward_varlen`

| Parameter | Dtype | Shape | Description |
| --- | --- | --- | --- |
| `x`, `r`, `k`, `v`, `g` | fp16 | `[N,C]` each | Packed source, R/K/V, and gate rows. |
| `r_k` | fp16 | `[C]` | Receptance/key residual scale. |
| `weight`, `bias` | fp16 | `[C]` each | Head-wise normalization affine parameters. |
| `batch_size` | int | scalar | Positive logical batch size; default `1`. |
| `max_seqlen` | int/None | scalar | Positive dispatch extent; defaults to `N`. |

#### `infer_tmix_vres_gate_forward_varlen`

| Parameter | Dtype | Shape | Description |
| --- | --- | --- | --- |
| `v` | fp16 | `[N,C]` | Current values. |
| `v_first` | fp16 | `[N,C]` | First-layer values. |
| `v0` | fp16 | `[C]` | Gate bias. |
| `v12` | fp16 | `[N,C]` | Per-token gate term. |

#### `infer_tmix_layer_norm_forward_varlen`

| Parameter | Dtype | Shape | Description |
| --- | --- | --- | --- |
| `x` | fp16 | `[N,C]` | Packed rows. |
| `weight`, `bias` | fp16 | `[C]` each | Layer-normalization affine parameters. |
| `eps` | float | scalar | Positive epsilon; default `1e-5`. |

#### `infer_tmix_add_layer_norm_forward_varlen`

| Parameter | Dtype | Shape | Description |
| --- | --- | --- | --- |
| `x`, `residual` | fp16 | `[N,C]` each | Rows to add and normalize. |
| `weight`, `bias` | fp16 | `[C]` each | Layer-normalization affine parameters. |
| `eps` | float | scalar | Epsilon; default `1e-5`. |
| `batch_size` | int/None | scalar | Optional positive batch hint, at most `N`. |

#### `infer_tmix_add_last_layer_norm_forward_varlen`

| Parameter | Dtype | Shape | Description |
| --- | --- | --- | --- |
| `x`, `residual` | fp16 | `[N,C]` each | Rows to add before final normalization. |
| `weight`, `bias` | fp16 | `[C]` each | Layer-normalization affine parameters. |
| `eps` | float | scalar | Epsilon; default `1e-5`. |

#### `infer_tmix_add_forward_varlen`

| Parameter | Dtype | Shape | Description |
| --- | --- | --- | --- |
| `x` | fp16 | `[N,C]` | Packed input rows. |
| `residual` | fp16 | `[N,C]` | Rows added elementwise to `x`. |

### Linear and low-rank parameters

#### `infer_tmix_linear_forward_varlen`

| Parameter | Dtype | Shape | Description |
| --- | --- | --- | --- |
| `x` | fp16 | `[M,K]` | Packed input matrix. |
| `weight` | fp16 | `[N,K]` or `[K,N]` | Projection weight in original or runtime-transposed layout. |
| `weight_is_transposed` | bool | scalar | Selects `[K,N]`; default `False`. |

#### `infer_tmix_linear_attention_c2c_forward_varlen`

| Parameter | Dtype | Shape | Description |
| --- | --- | --- | --- |
| `x` | fp16 | `[M,K]` | Packed attention input. |
| `weight` | fp16 | `[N,K]` | Original-layout base projection. |
| `lora_a` | fp16/None | `[R,K]` | Optional PEFT-layout rank-in weight; `1<=R<=512`. |
| `lora_b` | fp16/None | `[N,R]` | Optional rank-out weight; supplied with `lora_a`. |
| `lora_scale` | float | scalar | Finite float32-representable scale; default `1.0`. |

#### `infer_tmix_linear_ffn_key_forward_varlen`

| Parameter | Dtype | Shape | Description |
| --- | --- | --- | --- |
| `x` | fp16 | `[M,K]` | Packed FFN input. |
| `weight` | fp16 | `[N,K]` | Original-layout FFN-key projection. |

#### `infer_tmix_linear_t_forward_varlen`

| Parameter | Dtype | Shape | Description |
| --- | --- | --- | --- |
| `x` | fp16 | `[M,K]` | Packed input. |
| `weight_t` | fp16 | `[N,K]` | Projection consumed by the `linear_t` family. |

#### `infer_tmix_linear_t_tanh_forward_varlen`

| Parameter | Dtype | Shape | Description |
| --- | --- | --- | --- |
| `x` | fp16 | `[M,K]` | Input activated with tanh before projection. |
| `weight_t` | fp16 | `[N,K]` | Projection weight. |

#### `infer_tmix_linear_t_sigmoid_forward_varlen`

| Parameter | Dtype | Shape | Description |
| --- | --- | --- | --- |
| `x` | fp16 | `[M,K]` | Input activated with sigmoid before projection. |
| `weight_t` | fp16 | `[N,K]` | Projection weight. |

#### `infer_tmix_linear_act_tanh_forward_varlen`

| Parameter | Dtype | Shape | Description |
| --- | --- | --- | --- |
| `x` | fp16 | `[M,K]` | Input with an even total element count. |

#### `infer_tmix_linear_act_sigmoid_forward_varlen`

| Parameter | Dtype | Shape | Description |
| --- | --- | --- | --- |
| `x` | fp16 | `[M,K]` | Input with an even total element count. |

#### `infer_tmix_linear_t_vres_forward_varlen`

| Parameter | Dtype | Shape | Description |
| --- | --- | --- | --- |
| `x` | fp16 | `[M,K]` | Gate projection input. |
| `weight_t` | fp16 | `[N,K]` | Gate projection weight. |
| `v` | fp16 | `[M,N]` | Current values. |
| `v_first` | fp16 | `[M,N]` | First-layer values. |
| `v0` | fp16 | `[N]` | Value-residual gate bias. |

#### `infer_tmix_linear_rank_in_forward_varlen`

| Parameter | Dtype | Shape | Description |
| --- | --- | --- | --- |
| `x` | fp16 | `[M,K]` | Full-rank input. |
| `weight` | fp16/None | `[K,R]` | Optional runtime-layout rank-in weight. |
| `weight_t` | fp16/None | `[R,K]` | Optional original-layout rank-in weight. At least one weight is required. |

#### `infer_tmix_linear_rank_out_forward_varlen`

| Parameter | Dtype | Shape | Description |
| --- | --- | --- | --- |
| `x` | fp16 | `[M,R]` | Rank features. |
| `weight` | fp16/None | `[R,N]` | Optional runtime-layout rank-out weight. |
| `weight_t` | fp16/None | `[N,R]` | Optional original-layout rank-out weight. At least one weight is required. |

#### `infer_tmix_linear_rank_out_tanh_forward_varlen`

| Parameter | Dtype | Shape | Description |
| --- | --- | --- | --- |
| `x` | fp16 | `[M,R]` | Rank features activated with tanh. |
| `weight` | fp16/None | `[R,N]` | Optional runtime-layout rank-out weight. |
| `weight_t` | fp16/None | `[N,R]` | Optional original-layout rank-out weight. |

#### `infer_tmix_linear_rank_out_sigmoid_forward_varlen`

| Parameter | Dtype | Shape | Description |
| --- | --- | --- | --- |
| `x` | fp16 | `[M,R]` | Rank features activated with sigmoid. |
| `weight` | fp16/None | `[R,N]` | Optional runtime-layout rank-out weight. |
| `weight_t` | fp16/None | `[N,R]` | Optional original-layout rank-out weight. |

#### `infer_tmix_lowrank_in_forward_varlen`

| Parameter | Dtype | Shape | Description |
| --- | --- | --- | --- |
| `x_w`, `x_a`, `x_g` | fp16 | `[M,C]` each | W/A/G full-rank source rows. |
| `w1`, `a1`, `g1` | fp16/None | `[R,C]` each | Original-layout projections; positional compatibility is retained. |
| `w1_runtime`, `a1_runtime`, `g1_runtime` | fp16/None | `[C,R]` each | Keyword-only runtime layouts prepared by the caller. Each projection requires at least one layout. |

#### `infer_tmix_lowrank_wagv_in_forward_varlen`

| Parameter | Dtype | Shape | Description |
| --- | --- | --- | --- |
| `x_w`, `x_a`, `x_g`, `x_v` | fp16 | `[M,C]` each | W/A/G/V full-rank source rows. |
| `w1`, `a1`, `g1`, `v1` | fp16/None | `[R,C]` each | Original-layout projections. |
| `w1_runtime`, `a1_runtime`, `g1_runtime`, `v1_runtime` | fp16/None | `[C,R]` each | Keyword-only runtime layouts; no in-call transpose is performed. |

#### `infer_tmix_lowrank_out_forward_varlen`

| Parameter | Dtype | Shape | Description |
| --- | --- | --- | --- |
| `w1`, `a1`, `g1` | fp16 | `[M,R]` each | W/A/G rank features. |
| `w2`, `a2`, `g2` | fp16/None | `[C,R]` each | Original-layout second-stage projections. |
| `w2_runtime`, `a2_runtime`, `g2_runtime` | fp16/None | `[R,C]` each | Keyword-only runtime layouts prepared outside forward. |

#### `infer_tmix_lowrank_vres_forward_varlen`

| Parameter | Dtype | Shape | Description |
| --- | --- | --- | --- |
| `w1`, `a1`, `g1`, `v1` | fp16 | `[M,R]` each | W/A/G/V rank features. |
| `w2`, `a2`, `g2`, `v2` | fp16/None | `[C,R]` each | Original-layout second-stage projections. |
| `w2_runtime`, `a2_runtime`, `g2_runtime`, `v2_runtime` | fp16/None | `[R,C]` each | Keyword-only runtime layouts prepared outside forward. |
| `v`, `v_first` | fp16 | `[M,C]` each | Current and first-layer values. |
| `v0` | fp16 | `[C]` | Value-residual gate bias. |

### ChannelMix parameters

#### `infer_cmix_mix_forward_varlen`

| Parameter | Dtype | Shape | Description |
| --- | --- | --- | --- |
| `x` | fp16 | `[N,C]` | Packed token rows. |
| `x_k` | fp16 | `[C]` | ChannelMix interpolation vector. |
| `shift_state_pool` | fp16 | `[slots,C]` | Previous-token state, updated in place. |
| `cu_seqlens` | int32 | `[B+1]` | Packed sequence offsets. |
| `state_indices` | int32 | `[B]` | Shift-state slots. |
| `max_seqlen` | int/None | scalar | Optional maximum sequence length. |
| `validated_metadata` | object/None | opaque | Optional reusable metadata ticket. |

#### `infer_cmix_add_layer_norm_mix_forward_varlen`

| Parameter | Dtype | Shape | Description |
| --- | --- | --- | --- |
| `x`, `residual` | fp16 | `[B,4096]` each | Sequence-length-one rows to add and normalize. |
| `weight`, `bias`, `x_k` | fp16 | `[4096]` each | Normalization and mixing parameters. |
| `shift_state_pool` | fp16 | `[slots,4096]` | Previous-token state, updated in place. |
| `cu_seqlens` | int32 | `[B+1]` | Length-one sequence offsets. |
| `state_indices` | int32 | `[B]` | One state slot per row. |
| `max_seqlen` | int/None | scalar | Must resolve to `1`. |
| `eps` | float | scalar | Positive finite epsilon; default `1e-5`. |
| `validated_metadata` | object/None | opaque | Optional reusable metadata ticket. |

#### `infer_cmix_relu_square_forward_varlen`

| Parameter | Dtype | Shape | Description |
| --- | --- | --- | --- |
| `x` | fp16 | `[N,F]` | Dense FFN rows with even total element count. |

#### `infer_cmix_linear_ffn_down_forward_varlen`

| Parameter | Dtype | Shape | Description |
| --- | --- | --- | --- |
| `x` | fp16 | `[N,K]` | FFN hidden rows. |
| `weight` | fp16 | `[K,C]` | Runtime-layout down-projection weight. |

#### `infer_cmix_sparse_forward_varlen`

| Parameter | Dtype | Shape | Description |
| --- | --- | --- | --- |
| `x` | fp16 | `[N,C]` | Packed token rows. |
| `x_k` | fp16 | `[C]` | ChannelMix interpolation vector. |
| `key_fc` | fp16 | `[F,C]` | Sparse up-projection weights. |
| `value_fc` | fp16 | `[F,C]` | Sparse down-projection weights. |
| `shift_state_pool` | fp16 | `[slots,C]` | Previous-token state, updated in place. |
| `cu_seqlens` | int32 | `[B+1]` | Packed sequence offsets. |
| `state_indices` | int32 | `[B]` | Shift-state slots. |
| `max_seqlen` | int/None | scalar | Optional maximum sequence length. |
| `validated_metadata` | object/None | opaque | Optional reusable metadata ticket. |

#### `infer_cmix_sparse_up_forward_varlen`

| Parameter | Dtype | Shape | Description |
| --- | --- | --- | --- |
| `x` | fp16 | `[N,C]` | Packed token rows. |
| `x_k` | fp16 | `[C]` | ChannelMix interpolation vector. |
| `key_fc` | fp16 | `[F,C]` | Sparse up-projection weights. |
| `shift_state_pool` | fp16 | `[slots,C]` | Previous-token state, updated in place. |
| `cu_seqlens` | int32 | `[B+1]` | Packed sequence offsets. |
| `state_indices` | int32 | `[B]` | Shift-state slots. |
| `max_seqlen` | int/None | scalar | Optional maximum sequence length. |
| `validated_metadata` | object/None | opaque | Optional reusable metadata ticket. |

#### `infer_cmix_sparse_down_relu_forward_varlen`

| Parameter | Dtype | Shape | Description |
| --- | --- | --- | --- |
| `preact` | fp16 | `[N,F]` | Sparse up-projection output. |
| `value_fc` | fp16 | `[F,C]` | Down-projection weights; `C` is even. |
| `batch_size` | int/None | scalar | Optional positive logical batch size. |
| `max_seqlen` | int/None | scalar | Optional positive extent; supplied with `batch_size`. |

### Head parameters

#### `infer_head_linear_forward_varlen`

| Parameter | Dtype | Shape | Description |
| --- | --- | --- | --- |
| `x` | fp16 | `[N,C]` | Hidden rows. |
| `weight` | fp16 | `[vocab,C]` | Output-head weight. |

#### `infer_head_linear_all_forward_varlen`

| Parameter | Dtype | Shape | Description |
| --- | --- | --- | --- |
| `x` | fp16 | `[N,C]` | Hidden rows for all-token logits. |
| `weight` | fp16 | `[vocab,C]` | Output-head weight. |

#### `infer_head_linear_last_forward_varlen`

| Parameter | Dtype | Shape | Description |
| --- | --- | --- | --- |
| `x` | fp16 | `[B,C]` | Final hidden row per sequence. |
| `weight` | fp16 | `[vocab,C]` | Output-head weight. |
| `tokens_count` | int | scalar | Positive caller dispatch token count. |

#### `infer_head_last_norm_forward_varlen`

| Parameter | Dtype | Shape | Description |
| --- | --- | --- | --- |
| `x`, `residual` | fp16 | `[N,C]` each | Packed rows; `C` is even. |
| `last_indices` | int64 | `[B]` | Absolute packed-row indices to select. |
| `weight`, `bias` | fp16 | `[C]` each | Final normalization affine parameters. |
| `eps` | float | scalar | Numerical epsilon; default `1e-5`. |

### Sampling parameters

#### `setup_sampling_states`

| Parameter | Dtype | Shape | Description |
| --- | --- | --- | --- |
| `seed` | int | scalar | Seed used to initialize every Philox slot deterministically. |
| `num_slots` | int | scalar | Positive persistent RNG-slot count. |

#### `infer_sampling_temperature_topk_topp_forward_varlen`

| Parameter | Dtype | Shape | Description |
| --- | --- | --- | --- |
| `logits` | fp32 | `[capacity,V]` | Compact output-producing request rows; `V` is divisible by four. |
| `states` | int8 | `[num_slots,state_bytes]` | Philox pool returned by `setup_sampling_states`. |
| `slot_indices` | int32 | `[capacity]` | RNG slot selected by each row. |
| `temperature` | float/fp32 | scalar or `[capacity]` | Sampling temperature. |
| `top_k` | int/int32 | scalar or `[capacity]` | Top-k control. |
| `top_p` | float/fp32 | scalar or `[capacity]` | Nucleus threshold. |
| `sample_capacity` | int/None | scalar | Fixed row capacity; supplied with `num_active_samples`. |
| `num_active_samples` | int32/None | one element | Live active-prefix length, including during Graph replay. |

#### `infer_sampling_six_parameter_forward_varlen`

| Parameter | Dtype | Shape | Description |
| --- | --- | --- | --- |
| `logits` | fp32 | `[capacity,V]` | Compact output-producing request rows. |
| `penalties` | fp32 | `[num_slots,V]` | Per-slot additive penalty state, updated in place. |
| `states` | int8 | `[num_slots,state_bytes]` | Persistent Philox state pool. |
| `slot_indices` | int32 | `[capacity]` | RNG and penalty slot selected by each row. |
| `presence_penalty` | float/fp32 | scalar or `[capacity]` | Penalty applied when a token is present. |
| `frequency_penalty` | float/fp32 | scalar or `[capacity]` | Additive increment after sampling. |
| `penalty_decay` | float/fp32 | scalar or `[capacity]` | Existing-penalty decay. |
| `temperature` | float/fp32 | scalar or `[capacity]` | Sampling temperature. |
| `top_k` | int/int32 | scalar or `[capacity]` | Top-k control. |
| `top_p` | float/fp32 | scalar or `[capacity]` | Nucleus threshold. |
| `sample_capacity` | int/None | scalar | Fixed row capacity; supplied with `num_active_samples`. |
| `num_active_samples` | int32/None | one element | Live active-prefix length. |

### Pretraining parameters

#### `pretrain_recurrent_bf16`

| Parameter | Dtype | Shape | Description |
| --- | --- | --- | --- |
| `r` | bf16 | `[B,T,C]` | Receptance input. |
| `w` | bf16 | `[B,T,C]` | Canonical clampw-v3 decay input. |
| `k` | bf16 | `[B,T,C]` | Key input. |
| `v` | bf16 | `[B,T,C]` | Value input. |
| `a` | bf16 | `[B,T,C]` | Low-rank state-update factor. |
| `b` | bf16 | `[B,T,C]` | Low-rank state-update factor. |

`C` must be divisible by 64 and `T` by 16.

#### `pretrain_tmix_a_gate_bf16`

| Parameter | Dtype | Shape | Description |
| --- | --- | --- | --- |
| `a0` | bf16 | `[C]` | Broadcast gate bias. |
| `a12` | bf16 | `[B,T,C]` | Per-token gate term. |

#### `pretrain_tmix_vres_gate_bf16`

| Parameter | Dtype | Shape | Description |
| --- | --- | --- | --- |
| `value` | bf16 | `[B,T,C]` | Current values. |
| `first_value` | bf16 | `[B,T,C]` | First-layer values. |
| `v0` | bf16 | `[C]` | Broadcast gate bias. |
| `v12` | bf16 | `[B,T,C]` | Per-token gate term. |

#### `pretrain_tmix_mix6_bf16`

| Parameter | Dtype | Shape | Description |
| --- | --- | --- | --- |
| `x` | bf16 | `[B,T,C]` | Dense token input. |
| `x_r`, `x_w`, `x_k`, `x_v`, `x_a`, `x_g` | bf16 | `[C]` each | Six TimeMix coefficient vectors. |

#### `pretrain_tmix_kk_pre_bf16`

| Parameter | Dtype | Shape | Description |
| --- | --- | --- | --- |
| `key` | bf16 | `[B,T,C]` | Key input; `C` is divisible by 64. |
| `key_scale` | bf16 | `[C]` | Per-channel key scale. |
| `learning_rate` | bf16 | `[B,T,C]` | Per-token learning-rate input. |
| `learning_rate_scale` | bf16 | `[C]` | Per-channel learning-rate scale. |

#### `pretrain_tmix_lnx_rkvres_xg_bf16`

| Parameter | Dtype | Shape | Description |
| --- | --- | --- | --- |
| `x`, `r`, `k`, `v`, `g` | bf16 | `[B,T,C]` each | Dense source, R/K/V, and gate inputs. |
| `residual_scale` | bf16 | `[C/64,64]` | Per-head recurrent residual scale. |
| `weight`, `bias` | bf16 | `[C]` each | Head-wise normalization affine parameters. |

#### `pretrain_cmix_bf16`

| Parameter | Dtype | Shape | Description |
| --- | --- | --- | --- |
| `x` | bf16 | `[B,T,C]` | Dense ChannelMix input. |
| `x_k` | bf16 | `[C]` | Mixing coefficient vector. |
| `key_weight` | bf16 | `[4C,C]` | FFN up-projection weight. |
| `value_weight` | bf16 | `[C,4C]` | FFN down-projection weight. |

#### `pretrain_head_l2wrap_ce_bf16`

| Parameter | Dtype | Shape | Description |
| --- | --- | --- | --- |
| `hidden` | bf16 | `[B,T,C]` | Hidden states before the output head. |
| `weight` | bf16 | `[65536,C]` | Fixed-vocabulary output-head weight. |
| `targets` | int64 | `[B*T]` or equivalent | One target in `[0,65536)` per hidden row. |
| `chunk_rows` | int | scalar | Positive rows per memory-bounded head chunk; default `4096`. |

#### `pretrain_l2wrap_ce_bf16`

| Parameter | Dtype | Shape | Description |
| --- | --- | --- | --- |
| `logits` | bf16/fp32 | `[...,vocab]` | Raw logits with at least one row. |
| `targets` | int64 | `[...]` flattened per row | One target in `[0,vocab)` per logits row. |

### StateTune parameters

#### `statetune_recurrent_fp32io16`

| Parameter | Dtype | Shape | Description |
| --- | --- | --- | --- |
| `initial_state` | fp32 | `[B,H,D,D]` | Differentiable initial state; not mutated. |
| `sequence_chunk_offsets` | int32 | `[B+1]` | Range of chunks owned by each sequence. |
| `chunk_token_starts` | int32 | `[chunks]` | Inclusive packed-token start per chunk. |
| `chunk_token_ends` | int32 | `[chunks]` | Exclusive packed-token end per chunk. |
| `r` | fp16/bf16 | `[N,H,D]` | Receptance rows. |
| `decay_logits` | fp16/bf16 | `[N,H,D]` | Raw decay logits. |
| `k` | fp16/bf16 | `[N,H,D]` | Key rows. |
| `v` | fp16/bf16 | `[N,H,D]` | Value rows. |
| `a` | fp16/bf16 | `[N,H,D]` | Low-rank state-update factor. |
| `b` | fp16/bf16 | `[N,H,D]` | Low-rank state-update factor. |
| `scale` | float | scalar | Finite output scale; default `1.0`. |

`D` must be 64, 128, or 256.

### RL/Infctx parameters

#### `rl_infctx_chunk_fp32io16`

| Parameter | Dtype | Shape | Description |
| --- | --- | --- | --- |
| `r` | fp16/bf16 | `[N,H,D]` | Receptance rows. |
| `decay_logits` | fp16/bf16 | `[N,H,D]` | Raw decay logits. |
| `k` | fp16/bf16 | `[N,H,D]` | Key rows. |
| `v` | fp16/bf16 | `[N,H,D]` | Value rows. |
| `a` | fp16/bf16 | `[N,H,D]` | Low-rank state-update factor. |
| `b` | fp16/bf16 | `[N,H,D]` | Low-rank state-update factor. |
| `state_pool` | fp32/None | `[slots,H,D,D]` | Optional input pool; zeros are allocated when omitted. |
| `cu_seqlens` | int32 | `[B+1]` | Packed offsets from `0` to `N`. |
| `state_indices` | int32/None | `[B]` | Optional unique slots; defaults to `0..B-1`. |
| `chunk_size` | int | scalar | One of `16`, `32`, or `64`; default `16`. |
| `strategy` | str | scalar | `materialized` or `recompute`; default `recompute`. |
| `scale` | float | scalar | Finite output scale; default `1.0`. |
| `decay_bias` | fp16/bf16/None | `[H,D]` or `[H*D]` | Optional bias matching token dtype. |

`D` must be 64, 128, or 256. The function returns a new final pool and does
not mutate `state_pool`.

#### `rl_infctx_chunk_fp32io16_factor_recompute`

| Parameter | Dtype | Shape | Description |
| --- | --- | --- | --- |
| `*args` | same as base API | same as base API | Positional arguments forwarded to `rl_infctx_chunk_fp32io16`. |
| `**kwargs` | same as base API | same as base API | Keyword arguments forwarded after forcing `strategy='recompute'`. |
