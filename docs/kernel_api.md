# FlashRWKV2 Kernel API Reference

Complete, source-verified API reference for every public operator exported by `flashrwkv2` (root flat exports) and `flashrwkv2.mtp`.

All root-exported operators can be imported directly from `flashrwkv2` (e.g., `flashrwkv2.pretrain_tmix_wkv7_recurrent_bf16`) or from their owning submodule.

## Contents

- [Notation & Common Contracts](#notation--common-contracts)
- [Numerical Contract](#numerical-contract)
- [State and Packed Metadata](#state-and-packed-metadata)
  - [WKV7 State Handles](#wkv7-state-handles)
  - [Packed Metadata Tickets](#packed-metadata-tickets)
- [Inference Operators](#inference)
  - [Embedding LN0](#infer_embedding_ln0_forward_varlen)
  - [TMix PostNorm + TokenShift](#infer_tmix_postnorm_tokenshift_forward_varlen)
  - [TMix WKV Prepare](#infer_tmix_wkv_prepare_forward_varlen)
  - [TMix WKV7 Recurrence (FP16 & FP32)](#infer_tmix_wkv7_recurrent_fp16_forward_varlen--fp32io16)
  - [TMix Readout](#infer_tmix_readout_forward_varlen)
  - [CMix (ChannelMix)](#infer_cmix_forward_varlen)
  - [Final PostNorm](#infer_post_norm_output_forward_varlen)
  - [LM Head Linear](#infer_head_linear_all_forward_varlen)
  - [Six-Parameter Sampling](#setup_sampling_states--infer_sampling_six_parameter_forward_varlen)
  - [LightMTP DraftLoop](#lightmtp-draftloop)
- [Training Operators](#training)
  - [TMix TokenShift](#pretrain_tmix_tokenshift_bf16)
  - [TMix WKV Prepare](#pretrain_tmix_wkv_prepare_bf16)
  - [TMix WKV7 Recurrence](#pretrain_tmix_wkv7_recurrent_bf16)
  - [RL / Infinite-Context WKV7 Recurrence](#rl_infctx_tmix_wkv7_recurrent_bf16)
  - [TMix Readout](#pretrain_tmix_readout_bf16)
  - [CMix (ChannelMix)](#pretrain_cmix_bf16)
  - [LM Head Linear](#pretrain_head_linear_bf16)
  - [Chunked LM Head Cross-Entropy](#pretrain_head_ce_bf16)
  - [L2Wrap Cross-Entropy](#pretrain_l2wrap_ce_bf16)
  - [Chunked LM Head L2Wrap Cross-Entropy](#pretrain_head_l2wrap_ce_bf16)
  - [`torch.compile(fullgraph=True)` & Packed `cu_seqlens`](#torchcompilefullgraphtrue--packed-cu_seqlens)
- [Execution Configuration](#execution-configuration)

---

## Notation & Common Contracts

### Shape Symbols

| Symbol | Meaning | Constraints |
| --- | --- | --- |
| `B` | Batch size / number of active sequences | `B >= 1` |
| `T` | Dense sequence length | `T >= 1` |
| `N` | Packed total token count (`sum_i len_i`) | `N >= 1` |
| `C` | Model hidden dimension (`H * D`) | Divisible by `D` |
| `H` | Number of attention heads | `H >= 1` (D64 FP16 DeltaLog triggers for `H ∈ {40, 64}`) |
| `D` | Head width (`head_size`) | `D ∈ {64, 128, 256}` |
| `F` | CMix FFN intermediate width | `F >= 1` |
| `R` | Low-rank projection or LoRA/NoRA rank | `R >= 1` |
| `V` | Vocabulary size | `V >= 1` (`V % 4 == 0` and `V <= 1_048_576` for sampling; `V == 65536` for `pretrain_head_l2wrap_ce_bf16`) |
| `slots` | Capacity of a state pool | `slots >= 1` |

Unless noted otherwise, all tensor arguments must be **contiguous CUDA tensors on the same device**.

### Shared LoRA / NoRA Adapter Contract

Several projection operators accept optional LoRA/NoRA keyword arguments prefixed by projection name (e.g., `key_lora_a`, `output_lora_a`, or unprefixed `lora_a` on head operators):

| Argument | Shape | Dtype | Description |
| --- | --- | --- | --- |
| `*_lora_a` | `[R, in_features]` | `bfloat16` or `float32` | Down-projection matrix `A`. Must be paired with `*_lora_b`. |
| `*_lora_b` | `[out_features, R]` | `bfloat16` or `float32` | Up-projection matrix `B`. Must be paired with `*_lora_a`. |
| `*_lora_scale` | `float` (default `1.0`) | — | Finite scalar multiplier for the adapter branch. |
| `*_use_nora` | `bool` (default `False`) | — | If `True`, normalizes columns of `A` as `A / (norm(A, dim=0) + 1e-6)` in FP32 (forward and backward) before BF16 projection. |

- When the base weight has `requires_grad=False`, base weight gradients (`dW`) are skipped automatically while input and adapter gradients remain exact.

---

## Numerical Contract

- **Forward precision**: BF16 token activations and weights with FP32 accumulation and reductions across training, prefill, and decode.
- **Deterministic reductions**: Linear projections execute BF16 MMA in strictly increasing `K` order (no split-K). LayerNorm, head LayerNorm, and vocabulary reductions use fixed reduction trees.
- **BF16 rounding boundaries**: `res + x` rounds to BF16 before LayerNorm; projection, gate, TokenShift, and WKV outputs round to BF16 at identical boundaries across training and inference.
- **State precision**:
  - Training, RL rollout, evaluation, and sampling search use **FP32** WKV state (`[slots, H, D, D]` or `[B, H, D, D]`) with no narrowing between tokens.
  - General serving on SM89 (RTX 4090) and SM120 (PRO 6000) prefers **FP16** WKV state: BF16 token I/O, FP32 recurrence arithmetic, and per-token FP16 state rounding.
  - TokenShift state is BF16 `[slots, C]` (inference) or `[B, C]` (training).
  - LM head logits, selected token logprobs, and cross-entropy losses are **FP32**.

### WKV-7 Recurrence Equations

WKV takes raw decay logits `w` (if `decay_bias` is supplied, `w = bf16(fp32(w) + fp32(decay_bias))`). Retention is:

```text
retention = exp(-0.6065306597126334 * sigmoid(w))
```

With state layout `[K, V]` (`[D, D]` per head), each token updates:

```text
sa[v]       = sum_k a[k] * state[k, v]
state[k, v] = retention[k] * state[k, v] + b[k] * sa[v] + k[k] * v[v]
y[v]        = scale * sum_k r[k] * state[k, v]
```

### Backward Checkpointing

When gradients are enabled, WKV forward saves one FP32 state checkpoint every 16 tokens:

- **Direct D64 path** (`D == 64`, no live status, `N >= max(4096, 16 * B)`): replays per-token `sa`, optionally splits long sequences (`N > 4096` and `B * H < 128`) into 4096-token segments with a boundary adjoint pre-pass, and writes key/value gradients directly without atomics.
- **General path** (`D ∈ {64, 128, 256}`): replays a bounded 16-token shared-memory state history and accumulates key gradients via FP32 atomics.

---

## State and Packed Metadata

### WKV7 State Handles

```python
prepare_tmix_wkv7_recurrent_fp32io16_state_from_tensor(
    state: torch.Tensor,
) -> _TmixWkv7RecurrentState

prepare_tmix_wkv7_recurrent_fp16_state_from_tensor(
    state: torch.Tensor,
) -> _TmixWkv7RecurrentState
```

Bind an existing contiguous CUDA `[slots, H, D, D]` tensor (`float32` for `fp32io16`, `float16` for `fp16`, `D ∈ {64, 128, 256}`) into a managed state handle.

| Handle Member | Signature / Type | Semantics |
| --- | --- | --- |
| `memory_layout` | `property -> dict[str, int]` | Returns `base_bytes_per_slot`, `private_bytes_per_slot`, `bytes_per_slot`, `fixed_workspace_nbytes` (`0`), and `total_nbytes`. Use `bytes_per_slot` for concurrency budgeting. |
| `materialize_slots_(state_indices)` | `(int32 [B]) -> self` | Flushes pending DeltaLog updates for `state_indices` into the bound tensor. **Must be called before reading, copying, exporting, or rebinding the raw tensor directly.** |
| `clone()` | `() -> handle` | Materializes all slots and returns a deep copy of the entire pool. |
| `clone_slots(state_indices)` | `(int32 [B]) -> handle` | Materializes and clones only `state_indices` into a new handle of size `B`. |
| `copy_(source)` | `(handle) -> self` | Materializes `source` and copies the entire pool in place (shapes, dtype, device must match). |
| `copy_slots_(source, source_indices, destination_indices, *, validated_metadata=None)` | `(handle, int32 [B], int32 [B], *, object or None) -> self` | Copies selected slots, including overlapping same-pool copies. Active destination slots must be unique. A live ticket limits DeltaLog materialization and copying to the active sequence prefix. |
| `reset_slots_(state_indices)` | `(int32 [B]) -> self` | Zeros selected slots and clears their DeltaLog phase. |
| `zero_()` | `() -> self` | Zeros the entire pool and clears all DeltaLog phases. |

> **DeltaLog private memory (`fp16` handles on `D=64`)**:
> - SM89 RTX 4090 (128 SMs), `H=64`: up to 2 compact records/head/slot (**1,540 B/head**; 3.008 MiB per 32-layer sequence on top of 16 MiB base state).
> - Other `H ∈ {40, 64}` configs: 1 standard 5-vector record/head/slot (**1,284 B/head**; 2.508 MiB per 32-layer H64 sequence).
> - All other geometries and all `fp32io16` handles: **0 B** private memory.

For graph capture, prepare the live ticket on the same stream as the copy. Both index vectors have length `sequence_capacity` and describe the source and destination slots independently of the slots bound to the ticket. Only rows in `[0, num_active_sequences)` are validated and used to address the pools. Padding may repeat slot `0`. A valid live ticket supplies the active count on the GPU, so replay can change that count without a host synchronization.

### Packed Metadata Tickets

```python
prepare_tmix_wkv7_recurrent_metadata(
    cu_seqlens: torch.Tensor,
    state_indices: torch.Tensor,
    *,
    state_pool_size: int,
    total_tokens: int | None = None,
    max_seqlen: int | None = None,
    token_capacity: int | None = None,
    sequence_capacity: int | None = None,
    max_seqlen_capacity: int | None = None,
    num_active_tokens: torch.Tensor | None = None,
    num_active_sequences: torch.Tensor | None = None,
) -> object
```

Creates a validated metadata ticket shared across all stateful operators in a forward step (`tokenshift`, `wkv7`, `cmix`). Supports two mutually exclusive modes:

| Mode | Required Keyword Arguments | Behavior |
| --- | --- | --- |
| **Static eager** | `state_pool_size`, `total_tokens` (optional `max_seqlen`) | Validates bounds (`cu_seqlens` contiguous CUDA `int32 [B+1]` spanning `[0, total_tokens]`, `state_indices` unique in `[0, state_pool_size)`), snapshots buffers, and verifies identity/version/shape/stream on reuse. |
| **Live CUDA Graph** | `state_pool_size`, `token_capacity`, `sequence_capacity`, `max_seqlen_capacity`, `num_active_tokens` (1-element CUDA `int32`), `num_active_sequences` (1-element CUDA `int32`) | Keeps metadata and active counts live on device. Validates the active prefix and sorts active rows by descending length on every replay (outputs remain in input order). |

- **Live safety guarantees**: Zero-active warmup and invalid metadata never mutate any state pool. Inactive output rows (`row >= num_active_tokens`) receive `NaN` and must not be consumed downstream.

---

## Inference

The public TMix inference pipeline is fixed as `tokenshift → wkv_prepare → wkv7 → readout`, followed by `cmix`, `post_norm`, `head`, and `sampling` / `mtp`.

### `infer_embedding_ln0_forward_varlen`

```python
infer_embedding_ln0_forward_varlen(
    embedding: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    eps: float = 1e-05,
) -> torch.Tensor
```

Applies initial LayerNorm (`ln0`) to embeddings using FP32 statistics.

| Parameter | Shape | Dtype | Notes |
| --- | --- | --- | --- |
| `embedding` | `[..., C]` | `bfloat16` | Input token embeddings. |
| `weight`, `bias` | `[C]` | `bfloat16` | LayerNorm affine parameters. |
| `eps` | `float` | — | Variance epsilon (default `1e-5`). |

- **Returns**: `bfloat16` tensor with the same shape as `embedding`.

---

### `infer_tmix_postnorm_tokenshift_forward_varlen`

```python
infer_tmix_postnorm_tokenshift_forward_varlen(
    x: torch.Tensor,
    res: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    x_r: torch.Tensor,
    x_w: torch.Tensor,
    x_k: torch.Tensor,
    x_v: torch.Tensor,
    x_a: torch.Tensor,
    x_g: torch.Tensor,
    *,
    shift_state_pool: torch.Tensor,
    cu_seqlens: torch.Tensor,
    state_indices: torch.Tensor,
    max_seqlen: int | None = None,
    eps: float = 1e-05,
    validated_metadata: object | None = None,
    save_normalized: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]
```

Fuses `res_out = bf16(x + res)`, FP32-statistics LayerNorm, and six-way TokenShift on packed rows.

| Parameter | Shape | Dtype | Notes |
| --- | --- | --- | --- |
| `x`, `res` | `[N, C]` | `bfloat16` | Packed hidden and residual inputs. |
| `weight`, `bias` | `[C]` | `bfloat16` | LayerNorm affine parameters. |
| `x_r`, `x_w`, `x_k`, `x_v`, `x_a`, `x_g` | `[C]` | `bfloat16` | Six TokenShift interpolation vectors. |
| `shift_state_pool` | `[slots, C]` | `bfloat16` | **Mutated in place** for active slots with each sequence's final normalized row. |
| `cu_seqlens` | `[B + 1]` | `int32` | Packed sequence offsets. |
| `state_indices` | `[B]` | `int32` | Unique slot indices into `shift_state_pool`. |
| `max_seqlen` | `int \| None` | — | Optional upper bound on sequence length. |
| `validated_metadata` | `object \| None` | — | Optional ticket from `prepare_tmix_wkv7_recurrent_metadata`. |
| `save_normalized` | `[N, C]` or `None` | `bfloat16` | Optional non-aliasing buffer receiving each active token's post-LN shift state (for MTP rewind). |

- **Returns** (7-tuple of `bfloat16 [N, C]`): `(res_out, r, w, k, v, a, g)`.

---

### `infer_tmix_wkv_prepare_forward_varlen`

```python
infer_tmix_wkv_prepare_forward_varlen(
    x_r: torch.Tensor,
    x_w: torch.Tensor,
    x_k: torch.Tensor,
    x_v: torch.Tensor,
    x_a: torch.Tensor,
    x_g: torch.Tensor,
    receptance_weight: torch.Tensor,
    key_weight: torch.Tensor,
    value_weight: torch.Tensor,
    w1: torch.Tensor | None,
    a1: torch.Tensor | None,
    g1: torch.Tensor | None,
    v1: torch.Tensor | None,
    w2: torch.Tensor | None,
    a2: torch.Tensor | None,
    g2: torch.Tensor | None,
    v2: torch.Tensor | None,
    v0: torch.Tensor | None,
    k_k: torch.Tensor,
    a0: torch.Tensor,
    k_a: torch.Tensor,
    *,
    v_first: torch.Tensor | None = None,
    w1_runtime: torch.Tensor | None = None,
    a1_runtime: torch.Tensor | None = None,
    g1_runtime: torch.Tensor | None = None,
    v1_runtime: torch.Tensor | None = None,
    w2_runtime: torch.Tensor | None = None,
    a2_runtime: torch.Tensor | None = None,
    g2_runtime: torch.Tensor | None = None,
    v2_runtime: torch.Tensor | None = None,
    receptance_lora_a: torch.Tensor | None = None,
    receptance_lora_b: torch.Tensor | None = None,
    receptance_lora_scale: float = 1.0,
    key_lora_a: torch.Tensor | None = None,
    key_lora_b: torch.Tensor | None = None,
    key_lora_scale: float = 1.0,
    value_lora_a: torch.Tensor | None = None,
    value_lora_b: torch.Tensor | None = None,
    value_lora_scale: float = 1.0,
    receptance_use_nora: bool = False,
    key_use_nora: bool = False,
    value_use_nora: bool = False,
    head_size: int = 64,
    batch_size: int = 1,
    max_seqlen: int | None = None,
) -> tuple[torch.Tensor, ...]
```

Computes R/K/V dense projections (with optional LoRA/NoRA), low-rank W/A/G/V branches, Value Residual (`vres`), and normalized key / `recurrent_a` / `recurrent_b` preparation.

| Parameter Group | Shape | Dtype | Notes |
| --- | --- | --- | --- |
| `x_r, x_w, x_k, x_v, x_a, x_g` | `[N, C]` (or `[..., C]`) | `bfloat16` | Output from `tokenshift`. |
| `receptance_weight, key_weight, value_weight` | `[C, C]` | `bfloat16` | `[out, in]` layout. |
| `w1, a1, g1, v1` / `w2, a2, g2, v2` | `[R, C]` / `[C, R]` | `bfloat16` | Original `[out, in]` weights. May be `None` if corresponding `*_runtime` is provided. |
| `*_runtime` | `[C, R]` / `[R, C]` | `bfloat16` | Transposed `[in, out]` runtime views; enter the exact same MMA arithmetic without copying. |
| `v_first`, `v0`, `v1`/`v2` | `[N, C]`, `[C]` | `bfloat16` | Pass `v_first=None` on layer 0 (VRes skipped; returns `value` as `v_first`). On layers `> 0`, `v0`, `v1`, `v2` are required. |
| `k_k`, `a0`, `k_a` | `[C]` | `bfloat16` | Per-channel key normalization, in-context rate bias, and key scaling vectors. |
| `head_size` | `int` (`64/128/256`) | — | Head width `D`. (`batch_size` and `max_seqlen` are reserved compatibility kwargs.) |

> **Important (`w0` decay bias)**: Unlike `pretrain_tmix_wkv_prepare_bf16`, `infer_tmix_wkv_prepare_forward_varlen` does **not** add `w0`. Pass `decay_bias=w0` to `infer_tmix_wkv7_recurrent_*` to keep the exact BF16 rounding boundary.

- **Returns** (8-tuple of `bfloat16 [N, C]`): `(r, decay_logits, key, value, recurrent_a, recurrent_b, gate, v_first)`. Reshape the first 6 tensors to `[N, H, D]` before calling WKV7.

---

### `infer_tmix_wkv7_recurrent_fp16_forward_varlen` & `fp32io16`

```python
infer_tmix_wkv7_recurrent_fp16_forward_varlen(
    r: torch.Tensor | None,
    decay_logits: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    state: _TmixWkv7RecurrentState,
    cu_seqlens: torch.Tensor,
    state_indices: torch.Tensor,
    scale: float = 1.0,
    decay_bias: torch.Tensor | None = None,
    max_seqlen: int | None = None,
    validated_metadata: object | None = None,
    save: bool = False,
    boundaries: torch.Tensor | None = None,
    checkpoint_interval: int = 16,
) -> torch.Tensor | tuple[torch.Tensor | None, torch.Tensor] | None

infer_tmix_wkv7_recurrent_fp32io16_forward_varlen(
    r: torch.Tensor | None,
    decay_logits: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    state: _TmixWkv7RecurrentState,
    cu_seqlens: torch.Tensor,
    state_indices: torch.Tensor,
    scale: float = 1.0,
    decay_bias: torch.Tensor | None = None,
    max_seqlen: int | None = None,
    validated_metadata: object | None = None,
    save: bool = False,
    boundaries: torch.Tensor | None = None,
    checkpoint_interval: int = 16,
) -> torch.Tensor | tuple[torch.Tensor | None, torch.Tensor] | None
```

- Use **`fp16`** for general online serving on SM89/SM120 (per-token FP16 state rounding + automatic DeltaLog on D64 H40/H64 with balanced 4×16 key reduction `(s0 + s1) + (s2 + s3)`).
- Use **`fp32io16`** for RL rollout, evaluation, and sampling search (bitwise identical to training).

| Parameter | Shape | Dtype | Notes |
| --- | --- | --- | --- |
| `r` | `[N, H, D]` or `None` | `bfloat16` | Pass `None` for a state-only update. The kernel skips the readout reduction and output allocation. |
| `decay_logits, k, v, a, b` | `[N, H, D]` | `bfloat16` | Packed recurrence inputs (`D ∈ {64, 128, 256}`). |
| `state` | Handle (`[slots, H, D, D]`) | `float16` or `float32` | **Mutated in place** for active slots. Must match function precision. |
| `cu_seqlens` | `[B + 1]` | `int32` | Packed sequence offsets. |
| `state_indices` | `[B]` | `int32` | Unique slot indices into `state`. |
| `scale` | `float` (default `1.0`) | — | Finite output multiplier. |
| `decay_bias` | `[H, D]` or `[H * D]` or `None` | `bfloat16` or `float32` | Optional `w0` added in FP32 and rounded to BF16 before retention. |
| `validated_metadata` | `object \| None` | — | Optional ticket from `prepare_tmix_wkv7_recurrent_metadata`. |
| `save` | `bool` (default `False`) | — | If `True`, exports pre-token WKV state checkpoints every `checkpoint_interval` tokens. |
| `boundaries` | `[N // S + B, H, D, D]` or `None` | Matches `state` | Optional preallocated non-overlapping checkpoint buffer (allocated automatically if `None`). |
| `checkpoint_interval` | `int` (default `16`) | — | Positive token stride `S` for state export. |

- **Returns**:
  - `save=False` returns the BF16 output `[N, H, D]`, or `None` when `r=None`.
  - `save=True` returns `(output, boundaries)`. The first item is `None` when `r=None`. For sequence `i` with `start = cu_seqlens[i]`, token offset `j` divisible by `S` stores the **pre-token** state at `boundaries[start // S + i + j // S]`. Live graph replays only write active checkpoints.

For memory-bounded MTP verification, bind a single-layer `scratch_S` pool and reuse it across layers. Copy each layer's active main slots to scratch and verify the full draft with `save=False`. Once acceptance is known, pack each sequence's first `accepted_b + 1` verified inputs and run the same recurrence on that layer's main pool with `r=None, save=False`. Commit keeps the same recurrence, per-token state rounding and DeltaLog semantics as ordinary decode. TMix and CMix shift states are committed separately.

> **MTP Verifier Rewind (`checkpoint_interval=1`)**: If `m` out of `L` verified tokens are accepted for sequence `i`:
> - `m == 0`: restore WKV from `boundaries[start + i]`; restore TMix/CMix shift states from before the verifier step.
> - `0 < m < L`: restore WKV from `boundaries[start + i + m]`; restore TMix/CMix shift states from `save_normalized[start + m - 1]`.
> - `m == L`: keep the updated state pools.

---

### `infer_tmix_readout_forward_varlen`

```python
infer_tmix_readout_forward_varlen(
    wkv_output: torch.Tensor,
    receptance: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    r_k: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor,
    gate: torch.Tensor,
    output_weight: torch.Tensor,
    *,
    output_lora_a: torch.Tensor | None = None,
    output_lora_b: torch.Tensor | None = None,
    output_lora_scale: float = 1.0,
    head_size: int = 64,
    batch_size: int = 1,
    max_seqlen: int | None = None,
    output_use_nora: bool = False,
) -> torch.Tensor
```

Fuses per-head LayerNorm (GroupNorm), RKV residual `((r * k * r_k).sum(-1, keepdim=True) * v)`, output gating (`* gate`), and output linear projection (with optional LoRA/NoRA).

| Parameter | Shape | Dtype | Notes |
| --- | --- | --- | --- |
| `wkv_output` | `[N, C]` | `bfloat16` or `float32` | Output of WKV7 reshaped to 2D. |
| `receptance, key, value, gate` | `[N, C]` | `bfloat16` | Packed tensors from `wkv_prepare`. |
| `r_k` | `[H, D]` or `[C]` | `bfloat16` | Per-head residual scale. |
| `ln_weight, ln_bias` | `[C]` | `bfloat16` | Head LayerNorm affine parameters. |
| `output_weight` | `[C, C]` | `bfloat16` | `[out, in]` output projection weight. |
| `output_lora_*`, `output_use_nora` | — | — | See [LoRA / NoRA Contract](#shared-lora--nora-adapter-contract). |
| `head_size` | `int` (`64/128/256`) | — | Head width `D`. |

- **Returns**: `bfloat16` tensor `[N, C]`.

---

### `infer_cmix_forward_varlen`

```python
infer_cmix_forward_varlen(
    x: torch.Tensor,
    res: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    x_k: torch.Tensor,
    key_weight: torch.Tensor,
    value_weight: torch.Tensor,
    *,
    shift_state_pool: torch.Tensor,
    cu_seqlens: torch.Tensor,
    state_indices: torch.Tensor,
    max_seqlen: int | None = None,
    eps: float = 1e-05,
    validated_metadata: object | None = None,
    deterministic: bool = False,
    key_lora_a: torch.Tensor | None = None,
    key_lora_b: torch.Tensor | None = None,
    value_lora_a: torch.Tensor | None = None,
    value_lora_b: torch.Tensor | None = None,
    key_lora_scale: float = 1.0,
    value_lora_scale: float = 1.0,
    key_use_nora: bool = False,
    value_use_nora: bool = False,
    save_normalized: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]
```

Full ChannelMix island: `res/ln` → 1-way TokenShift → key projection → ReLU² → value projection.

| Parameter | Shape | Dtype | Notes |
| --- | --- | --- | --- |
| `x, res` | `[N, C]` | `bfloat16` | Packed hidden and residual inputs. |
| `weight, bias, x_k` | `[C]` | `bfloat16` | LayerNorm parameters and TokenShift mix vector. |
| `key_weight` | `[F, C]` | `bfloat16` | Key projection weight. |
| `value_weight` | **`[F, C]`** | `bfloat16` | **Runtime layout** (transposed relative to training's `[C, F]`). |
| `shift_state_pool` | `[slots, C]` | `bfloat16` | **Mutated in place** for active slots. Must not alias projection weights/adapters. |
| `cu_seqlens`, `state_indices`, `validated_metadata`, `save_normalized` | — | — | Same contract as `infer_tmix_postnorm_tokenshift_forward_varlen`. |
| `key_lora_*`, `value_lora_*` | — | — | See [LoRA / NoRA Contract](#shared-lora--nora-adapter-contract) (`value_lora_a: [R, F]`, `value_lora_b: [C, R]`). |

- **Returns** (2-tuple of `bfloat16 [N, C]`): `(res_out, cmix_out)`.

---

### `infer_post_norm_output_forward_varlen`

```python
infer_post_norm_output_forward_varlen(
    x: torch.Tensor,
    res: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    eps: float = 1e-05,
) -> torch.Tensor
```

Computes `ln(bf16(x + res))` with FP32 statistics for the final model output layer.

- **Inputs**: `x, res: bfloat16 [..., C]`; `weight, bias: bfloat16 [C]`.
- **Returns**: `bfloat16 [..., C]`.

---

### `infer_head_linear_all_forward_varlen`

```python
infer_head_linear_all_forward_varlen(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    lora_a: torch.Tensor | None = None,
    lora_b: torch.Tensor | None = None,
    lora_scale: float = 1.0,
    use_nora: bool = False,
) -> torch.Tensor
```

Projects packed BF16 hidden states to FP32 vocabulary logits using the exact training forward.

- **Inputs**: `x: bfloat16 [N, C]`; `weight: bfloat16 [V, C]`; optional `lora_a [R, C]`, `lora_b [V, R]`.
- **Returns**: `float32 [N, V]`.

---

### `setup_sampling_states` & `infer_sampling_six_parameter_forward_varlen`

```python
setup_sampling_states(seed: int, num_slots: int) -> torch.Tensor

infer_sampling_six_parameter_forward_varlen(
    logits: torch.Tensor,
    penalties: torch.Tensor,
    states: torch.Tensor,
    slot_indices: torch.Tensor,
    *,
    presence_penalty: float | torch.Tensor = 0.0,
    frequency_penalty: float | torch.Tensor = 0.0,
    penalty_decay: float | torch.Tensor = 0.996,
    temperature: float | torch.Tensor = 1.0,
    top_k: int | torch.Tensor = -1,
    top_p: float | torch.Tensor = 1.0,
    sample_capacity: int | None = None,
    num_active_samples: torch.Tensor | None = None,
) -> torch.Tensor
```

Persistent Philox RNG initialization and fused six-parameter sampling with additive per-slot penalties.

| Parameter | Shape | Dtype | Notes |
| --- | --- | --- | --- |
| `logits` | `[B, V]` | `float32` | `V` must satisfy `0 < V <= 1_048_576` and `V % 4 == 0`. |
| `penalties` | `[num_slots, V]` | `float32` | **Mutated in place** for active slots (decayed by `penalty_decay` then incremented on sampled token). |
| `states` | `[num_slots, ...]` | `int8` | **Mutated in place**; allocated by `setup_sampling_states(seed, num_slots)`. |
| `slot_indices` | `[B]` | `int32` | Unique in-range slot indices. |
| `presence_penalty`, `frequency_penalty`, `penalty_decay`, `temperature`, `top_p` | scalar `float` or `[B]` | `float32` | Per-request or broadcast controls. **`frequency_penalty` is an additive increment** (not a multiplicative repetition penalty). |
| `top_k` | scalar `int` or `[B]` | `int32` | `<= 0` disables top-k filtering. |
| `sample_capacity`, `num_active_samples` | `int`, 1-element `[1]` | `int32` | Optional pair for live CUDA Graph capture (`sample_capacity == B`). Must be provided together. |

- **Returns**: Sampled token IDs tensor (`[B]`, CUDA `int32`).

---

### LightMTP `DraftLoop`

Imported from `flashrwkv2.mtp`:

```python
from flashrwkv2.mtp import DraftLoop

DraftLoop(
    state: Mapping[str, torch.Tensor],
    lm_head_weight: torch.Tensor,
    *,
    depth: int,
    eps: float = 1e-05,
    lora_a: torch.Tensor | None = None,
    lora_b: torch.Tensor | None = None,
    lora_scale: float = 1.0,
    use_nora: bool = False,
)
DraftLoop.from_state_dict(
    state: Mapping[str, torch.Tensor],
    lm_head_weight: torch.Tensor,
    *,
    depth: int,
    eps: float = 1e-05,
    lora_a: torch.Tensor | None = None,
    lora_b: torch.Tensor | None = None,
    lora_scale: float = 1.0,
    use_nora: bool = False,
) -> DraftLoop
```

Batched LightMTP speculative drafting supports eager execution and independent CUDA Graphs for each captured batch size.

Use the canonical `mtp.*` keys or replace `mtp.` with `block.` for an existing sidecar. Required weights must share one prefix. If a mapping has both forms, a complete legacy sidecar takes precedence. The block weights convert to contiguous BF16 on `lm_head_weight.device`.

| Key | Shape |
| --- | --- |
| `mtp.proj.weight` | `[C, 2 * C]` |
| `mtp.gate.weight`, `mtp.up.weight` | `[F, C]` |
| `mtp.down.weight` | `[C, F]` |
| `mtp.norm_{e,h,mlp,out}.{weight,bias}` | `[C]` |

`lm_head_weight` has shape `[V, C]` and reuses its storage when already contiguous CUDA BF16. The optional LoRA/NoRA arguments follow `infer_head_linear_all_forward_varlen`. Each step uses its FP32 `[B, V]` logits for argmax and releases them before the next step. Only the `[B, depth]` IDs accumulate across steps.

| Method | Signature | Semantics |
| --- | --- | --- |
| `forward` | `(hidden, token_ids, embed) -> int64 [B, depth]` | Eagerly drafts from CUDA BF16 `hidden[B, C]` and CUDA int64 `token_ids[B]`, with `B >= 1`. For `B=1`, `hidden[C]` and scalar `token_ids` also work. The embedding callable maps CUDA int64 `[B]` to CUDA BF16 `[B, C]`. |
| `capture` | `(embed, batch_size=1) -> None` | Warms up and captures a CUDA Graph for that batch size and embedding. Capture again after moving devices or loading module weights. |
| `replay` | `(hidden, token_ids) -> int64 [B, depth]` | Stages inputs and replays the graph captured for their batch size on the current stream. |

Calls to `forward` and `replay` reuse the same output tensor for a given batch size. Call `.clone()` to retain the result across calls of that size and invoke each instance serially.

---

## Training

All training operators support autograd, frozen base weights, and `torch.compile(fullgraph=True)`.

### `pretrain_tmix_tokenshift_bf16`

```python
pretrain_tmix_tokenshift_bf16(
    x: torch.Tensor,
    x_r: torch.Tensor,
    x_w: torch.Tensor,
    x_k: torch.Tensor,
    x_v: torch.Tensor,
    x_a: torch.Tensor,
    x_g: torch.Tensor,
    *,
    initial_shift: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    return_final_state: bool = False,
) -> tuple[torch.Tensor, ...]
```

Differentiable six-way TokenShift supporting dense (`[B, T, C]`, `cu_seqlens=None`) or packed (`[N, C]`, `cu_seqlens: int32 [B + 1]`) inputs.

- **`initial_shift`**: Optional differentiable BF16 `[B, C]` (defaults to zeros).
- **Returns**: `(r, w, k, v, a, g)` matching `x.shape`; appends `final_shift: bfloat16 [B, C]` when `return_final_state=True`.

---

### `pretrain_tmix_wkv_prepare_bf16`

```python
pretrain_tmix_wkv_prepare_bf16(
    x_r: torch.Tensor,
    x_w: torch.Tensor,
    x_k: torch.Tensor,
    x_v: torch.Tensor,
    x_a: torch.Tensor,
    x_g: torch.Tensor,
    receptance_weight: torch.Tensor,
    key_weight: torch.Tensor,
    value_weight: torch.Tensor,
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    a0: torch.Tensor,
    a1: torch.Tensor,
    a2: torch.Tensor,
    g1: torch.Tensor,
    g2: torch.Tensor,
    k_k: torch.Tensor,
    k_a: torch.Tensor,
    *,
    v_first: torch.Tensor | None = None,
    v0: torch.Tensor | None = None,
    v1: torch.Tensor | None = None,
    v2: torch.Tensor | None = None,
    receptance_lora_a: torch.Tensor | None = None,
    receptance_lora_b: torch.Tensor | None = None,
    key_lora_a: torch.Tensor | None = None,
    key_lora_b: torch.Tensor | None = None,
    value_lora_a: torch.Tensor | None = None,
    value_lora_b: torch.Tensor | None = None,
    receptance_lora_scale: float = 1.0,
    key_lora_scale: float = 1.0,
    value_lora_scale: float = 1.0,
    receptance_use_nora: bool = False,
    key_use_nora: bool = False,
    value_use_nora: bool = False,
    head_size: int = 64,
) -> tuple[torch.Tensor, ...]
```

Differentiable WKV preparation matching inference bitwise at every BF16 edge.

- **Inputs**: `x_*` have shape `[B, T, C]` or `[N, C]` (`bfloat16`); dense R/K/V weights are `[C, C]`; **low-rank weights use training `[in, out]` layout** (`w1, a1, g1, v1: [C, R]`; `w2, a2, g2, v2: [R, C]`); `w0, a0, v0, k_k, k_a: [C]`.
- **Returns** (8-tuple matching `x_r.shape`): `(r, decay_logits, key, value, recurrent_a, recurrent_b, gate, v_first)` with `w0` already added into `decay_logits`.

---

### `pretrain_tmix_wkv7_recurrent_bf16`

```python
pretrain_tmix_wkv7_recurrent_bf16(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    head_size: int = 64,
    initial_state: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    return_final_state: bool = False,
    scale: float = 1.0,
    decay_bias: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]
```

Differentiable RWKV-7 recurrence with functional (non-mutated) FP32 state.

| Mode | Input Shapes (`r, w, k, v, a, b`) | `cu_seqlens` | `initial_state` |
| --- | --- | --- | --- |
| **Dense** | `bfloat16 [B, T, C]` | `None` | Optional `float32 [B, H, D, D]` (`H = C // head_size`) |
| **Packed** | `bfloat16 [N, H, D]` | `int32 [B + 1]` | Optional `float32 [B, H, D, D]` |

- **Returns**: `bfloat16` output matching `r.shape`, or `(output, final_state)` (`float32 [B, H, D, D]`) when `return_final_state=True`.

---

### `rl_infctx_tmix_wkv7_recurrent_bf16`

```python
rl_infctx_tmix_wkv7_recurrent_bf16(
    r: torch.Tensor,
    decay_logits: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    state_pool: torch.Tensor,
    cu_seqlens: torch.Tensor,
    state_indices: torch.Tensor | None = None,
    scale: float = 1.0,
    decay_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]
```

Differentiable packed recurrence over a slot-indexed FP32 state pool for RL and infinite-context training.

- **Inputs**: `r, decay_logits, k, v, a, b: bfloat16 [N, H, D]`; `state_pool: float32 [slots, H, D, D]`; `cu_seqlens: int32 [B + 1]`; optional unique `state_indices: int32 [B]`.
- **Returns**: `(output, final_state_pool)` (`bfloat16 [N, H, D]`, `float32 [slots, H, D, D]`). Unselected slots remain unchanged; gradients flow through inputs and state pool.

---

### `pretrain_tmix_readout_bf16`

```python
pretrain_tmix_readout_bf16(
    x: torch.Tensor,
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    residual_scale: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    g: torch.Tensor,
    *,
    head_size: int = 64,
    output_weight: torch.Tensor | None = None,
    output_lora_a: torch.Tensor | None = None,
    output_lora_b: torch.Tensor | None = None,
    output_lora_scale: float = 1.0,
    output_use_nora: bool = False,
) -> torch.Tensor
```

Differentiable head LayerNorm, RKV residual, output gate, and optional output projection.

- **Inputs**: `x: bfloat16/float32 [B, T, C]`; `r, k, v, g: bfloat16 [B, T, C]`; `residual_scale: bfloat16 [H, D]` or `[C]`; `weight, bias: bfloat16 [C]`; optional `output_weight: bfloat16 [C, C]` (required if `output_lora_*` is passed).
- **Returns**: `bfloat16 [B, T, C]`.

---

### `pretrain_cmix_bf16`

```python
pretrain_cmix_bf16(
    x: torch.Tensor,
    x_k: torch.Tensor,
    key_weight: torch.Tensor,
    value_weight: torch.Tensor,
    *,
    initial_shift: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    return_final_state: bool = False,
    key_lora_a: torch.Tensor | None = None,
    key_lora_b: torch.Tensor | None = None,
    value_lora_a: torch.Tensor | None = None,
    value_lora_b: torch.Tensor | None = None,
    key_lora_scale: float = 1.0,
    value_lora_scale: float = 1.0,
    key_use_nora: bool = False,
    value_use_nora: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]
```

Differentiable ChannelMix on dense (`[B, T, C]`) or packed (`[N, C]` with `cu_seqlens`) inputs.

- **Weights**: `key_weight: bfloat16 [F, C]`; **`value_weight: bfloat16 [C, F]`** (training layout, transposed relative to inference).
- **Returns**: `bfloat16` matching `x.shape`, or `(output, final_shift)` (`bfloat16 [B, C]`) when `return_final_state=True`.

---

### `pretrain_head_linear_bf16`

```python
pretrain_head_linear_bf16(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    lora_a: torch.Tensor | None = None,
    lora_b: torch.Tensor | None = None,
    lora_scale: float = 1.0,
    use_nora: bool = False,
) -> torch.Tensor
```

Differentiable projection from `x: bfloat16 [..., C]` and `weight: bfloat16 [V, C]` to `float32 [..., V]` logits.

---

### `pretrain_head_ce_bf16`

```python
pretrain_head_ce_bf16(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    *,
    chunk_rows: int = 128,
    backward_chunk_rows: int | None = None,
    lora_a: torch.Tensor | None = None,
    lora_b: torch.Tensor | None = None,
    lora_scale: float = 1.0,
    use_nora: bool = False,
) -> torch.Tensor
```

Memory-bounded chunked LM head projection + per-token FP32 cross-entropy loss.

| Parameter | Shape | Dtype | Notes |
| --- | --- | --- | --- |
| `hidden` | `[B, T, C]` | `bfloat16` | Final normalized hidden states. |
| `weight` | `[V, C]` | `bfloat16` | LM head weight (arbitrary `V >= 1`). |
| `targets` | `B * T` elements | `int64` | Values in `[0, V)` or `-100` (ignored: zero loss and zero gradient). |
| `chunk_rows` | `int > 0` (default `128`) | — | Max rows materialized per forward chunk. |
| `backward_chunk_rows` | `int > 0 \| None` | — | Max rows per backward chunk (defaults to `chunk_rows`). |

- **Returns**: `float32` per-token cross-entropy matching `targets.shape`.

---

### `pretrain_l2wrap_ce_bf16`

```python
pretrain_l2wrap_ce_bf16(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor
```

Computes scalar FP32 mean cross-entropy with the `train_temp` L2Wrap logit regularization term in backward.

- **Inputs**: `logits: bfloat16/float32 [..., V]`; `targets: int64` with one entry in `[0, V)` per row.
- **Returns**: Scalar `float32` mean cross-entropy loss.

---

### `pretrain_head_l2wrap_ce_bf16`

```python
pretrain_head_l2wrap_ce_bf16(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    *,
    chunk_rows: int = 4096,
    lora_a: torch.Tensor | None = None,
    lora_b: torch.Tensor | None = None,
    lora_scale: float = 1.0,
    use_nora: bool = False,
) -> torch.Tensor
```

Memory-bounded chunked LM head projection + `train_temp` L2Wrap cross-entropy for RWKV's 65,536-token vocabulary.

- **Inputs**: `hidden: bfloat16 [B, T, C]`; `weight: bfloat16 [65536, C]`; `targets: int64` in `[0, 65536)` with `B * T` elements; `chunk_rows: int > 0`.
- **Returns**: Scalar `float32` mean cross-entropy loss.

---

### `torch.compile(fullgraph=True)` & Packed `cu_seqlens`

- All training operators register custom ops (`torch.library.custom_op`) invoking the identical TileLang forward/VJP kernels. To preserve BF16 intermediate rounding bitwise under Inductor:

  ```python
  compiled_model = torch.compile(
      model, fullgraph=True, options={"emulate_precision_casts": True},
  )
  ```

- Packed `cu_seqlens` in training must be contiguous CUDA `int32 [B + 1]`, starting at `0`, non-decreasing, and ending at total tokens `N`. Violations fail via asynchronous device assertion (`torch._assert_async`), which invalidates the CUDA context and requires restarting the process.

---

## Execution Configuration

- **Deterministic Hardware Tiles**: SM89 (RTX 4090, 128 SMs) and SM120 (PRO 6000, 188 SMs) select projection tiles and D64 WKV prefetch settings deterministically from geometry, SM count, and shared memory capacity.
- **Bitwise-Guarded Tuning**: Every benchmark under `benchmarks/` supports `--tune`, verifying candidate outputs and states bitwise against the canonical kernel before measuring CUDA Graph replay latency.
- **Caches & Prewarming**:
  - `FLASHRWKV2_CACHE_DIR`: Native extension cache root (default `~/.cache/flashrwkv2`).
  - `TILELANG_CACHE_DIR`: TileLang compiled binary cache root.
  - `python -m flashrwkv2.compile`: Prewarms the metadata/sampling native extension and emits JSON `{"status": "compiled" | "cached", "target": "smXX", "cache_key": "...", "library": "..."}`.

See [Quickstart & Recipes](usr/quickstart.md) for end-to-end recipes, the [Serving & State Management Guide](usr/serving.md) for CUDA Graph patterns, and [Release Acceptance](dev/release.md) for release validation and publication.
