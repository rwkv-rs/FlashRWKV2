# Quickstart & Recipes

This guide walks through the most common **FlashRWKV2** integration patterns: training a full RWKV-7 block, LoRA/NoRA fine-tuning, RL rollout & infinite-context training, LightMTP speculative decoding, and running benchmarks.

---

## 1. Environment & Cache Setup

Install the editable project and dev dependencies into a project-local `./.venv` managed by `uv`:

```bash
uv sync --group dev
```

Optional — prewarm the small native extension (metadata validation & sampling) before launching workers:

```bash
./.venv/bin/python -m flashrwkv2.compile
```

This prints JSON with `status` (`"compiled"` or `"cached"`), `target`, `cache_key`, and `library`. Set `FLASHRWKV2_CACHE_DIR` to customize the native extension cache path (defaults to `~/.cache/flashrwkv2`).

---

## 2. Full RWKV-7 Training Block

All training operators take contiguous CUDA **BF16** activations/weights, accumulate in **FP32**, and maintain differentiable **FP32** WKV state. Note that the TMix output is added into the residual stream before computing LN2 for CMix (matching [`tests/tmix/wkv7/test_block.py`](../../tests/tmix/wkv7/test_block.py) and upstream `Block.forward`):

```python
import torch
import torch.nn.functional as F
import flashrwkv2

def _bf16_layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return F.layer_norm(x.float(), (x.shape[-1],), weight.float(), bias.float(), eps).to(torch.bfloat16)

def rwkv7_block_train(
    x,                  # [B, T, C] BF16 residual stream entering the block
    tmix_params,
    cmix_params,
    *,
    head_size: int = 64,
    v_first=None,
    initial_wkv_state=None,
    initial_tmix_shift=None,
    initial_cmix_shift=None,
):
    # 1. LN1 + TMix TokenShift (6-way)
    x_ln1 = _bf16_layer_norm(x, tmix_params["ln1_w"], tmix_params["ln1_b"])
    x_r, x_w, x_k, x_v, x_a, x_g, final_tmix_shift = (
        flashrwkv2.pretrain_tmix_tokenshift_bf16(
            x_ln1,
            tmix_params["x_r"], tmix_params["x_w"], tmix_params["x_k"],
            tmix_params["x_v"], tmix_params["x_a"], tmix_params["x_g"],
            initial_shift=initial_tmix_shift,
            return_final_state=True,
        )
    )

    # 2. TMix WKV Prepare (R/K/V + low-rank W/A/G/V + VRes + KK/A)
    r, w, k, v, a, b, g, v_first = flashrwkv2.pretrain_tmix_wkv_prepare_bf16(
        x_r, x_w, x_k, x_v, x_a, x_g,
        tmix_params["r_weight"], tmix_params["k_weight"], tmix_params["v_weight"],
        tmix_params["w0"], tmix_params["w1"], tmix_params["w2"],
        tmix_params["a0"], tmix_params["a1"], tmix_params["a2"],
        tmix_params["g1"], tmix_params["g2"],
        tmix_params["k_k"], tmix_params["k_a"],
        v_first=v_first,
        v0=tmix_params.get("v0"),
        v1=tmix_params.get("v1"),
        v2=tmix_params.get("v2"),
        head_size=head_size,
    )

    # 3. TMix WKV-7 Recurrence (differentiable FP32 state)
    wkv_out, final_wkv_state = flashrwkv2.pretrain_tmix_wkv7_recurrent_bf16(
        r, w, k, v, a, b,
        head_size=head_size,
        initial_state=initial_wkv_state,
        return_final_state=True,
    )

    # 4. TMix Readout + residual update (connects TMix gradients into CMix)
    tmix_out = flashrwkv2.pretrain_tmix_readout_bf16(
        wkv_out, r, k, v,
        tmix_params["r_k"],
        tmix_params["ln_x_weight"], tmix_params["ln_x_bias"],
        g,
        head_size=head_size,
        output_weight=tmix_params["o_weight"],
    )
    x = (x + tmix_out).to(torch.bfloat16)

    # 5. LN2 + CMix (TokenShift + key proj + ReLU² + value proj) + residual update
    x_ln2 = _bf16_layer_norm(x, cmix_params["ln2_w"], cmix_params["ln2_b"])
    cmix_out, final_cmix_shift = flashrwkv2.pretrain_cmix_bf16(
        x_ln2,
        cmix_params["x_k"],
        cmix_params["key_weight"],    # [F, C]
        cmix_params["value_weight"],  # [C, F] (training layout)
        initial_shift=initial_cmix_shift,
        return_final_state=True,
    )
    x = (x + cmix_out).to(torch.bfloat16)
    return x, v_first, final_wkv_state, final_tmix_shift, final_cmix_shift
```

> For bitwise-matched fused residual+LN across training and packed inference, see `train_layer` in [`tests/tmix/wkv7/test_block.py`](../../tests/tmix/wkv7/test_block.py).

### Enabling `torch.compile(fullgraph=True)`

Preserve the BF16 intermediate rounding contract with Inductor's `emulate_precision_casts` option so compiled gradients match eager bitwise:

```python
compiled_model = torch.compile(
    model,
    fullgraph=True,
    options={"emulate_precision_casts": True},
)
```

For CUDA Graph training capture: warm forward and backward on the capture stream using the exact shapes, weight layouts, and `requires_grad` flags, then restore state and zero gradients before capturing.

### Chunked LM Head Loss (Memory-Bounded)

To avoid materializing full `[B*T, V]` FP32 logits during training, use `pretrain_head_ce_bf16` (per-token cross-entropy with `-100` ignore index) or `pretrain_head_l2wrap_ce_bf16` (with `train_temp` L2Wrap regularization):

```python
per_token_loss = flashrwkv2.pretrain_head_ce_bf16(
    hidden,             # [B, T, C] BF16
    lm_head_weight,     # [V, C] BF16
    targets,            # [B, T] int64 (-100 ignored)
    chunk_rows=128,
    backward_chunk_rows=128,
)
```

---

## 3. First-Class LoRA & NoRA Fine-Tuning

LoRA (`A: [R, K]`, `B: [O, R]`) and NoRA (`A / (norm(A, dim=0) + 1e-6)` in FP32) are built directly into TMix, CMix, and LM Head operators. When base weights have `requires_grad=False`, FlashRWKV2 skips `dW` while keeping input and adapter gradients intact.

| Operator | Adapter Keyword Prefix |
| --- | --- |
| `pretrain_tmix_wkv_prepare_bf16` / `infer_tmix_wkv_prepare_forward_varlen` | `receptance_*`, `key_*`, `value_*` |
| `pretrain_tmix_readout_bf16` / `infer_tmix_readout_forward_varlen` | `output_*` |
| `pretrain_cmix_bf16` / `infer_cmix_forward_varlen` | `key_*`, `value_*` |
| `pretrain_head_linear_bf16` / `pretrain_head_ce_bf16` / `pretrain_head_l2wrap_ce_bf16` | `lora_a`, `lora_b`, `lora_scale`, `use_nora` |

Example with frozen base weights and NoRA on the LM head:

```python
lm_head_weight.requires_grad_(False)
loss = flashrwkv2.pretrain_head_ce_bf16(
    hidden,
    lm_head_weight,
    targets,
    lora_a=head_lora_a,   # [R, C], BF16 or FP32
    lora_b=head_lora_b,   # [V, R], BF16 or FP32
    lora_scale=alpha / rank,
    use_nora=True,
)
```

---

## 4. RL Rollout & Infinite-Context Training

RL algorithms (GRPO / PPO / Reinforce++) are sensitive to train–rollout logprob mismatch. FlashRWKV2 guarantees **bitwise-identical** outputs, FP32 states, and selected logprobs between rollout and learner:

1. **Rollout / Evaluation**: Use `infer_tmix_wkv7_recurrent_fp32io16_forward_varlen` with an FP32 state handle from `prepare_tmix_wkv7_recurrent_fp32io16_state_from_tensor`.
2. **Learner (Packed + Slot-Indexed State Pool)**: Use `rl_infctx_tmix_wkv7_recurrent_bf16`:

```python
output, updated_pool = flashrwkv2.rl_infctx_tmix_wkv7_recurrent_bf16(
    r, decay_logits, k, v, a, b,  # packed [N, H, D] BF16
    state_pool=fp32_pool,         # [slots, H, D, D] FP32 (differentiable)
    cu_seqlens=cu_seqlens,        # [B + 1] CUDA int32
    state_indices=state_indices,  # [B] unique CUDA int32 (optional)
)
```

Unselected slots in `state_pool` are preserved unmodified, and gradients flow through both the packed token inputs and the selected initial/final states.

---

## 5. LightMTP Speculative Decoding

### Drafting (`DraftLoop`)

```python
from flashrwkv2.mtp import DraftLoop

# sidecar_weights holds `block.proj.weight`, `block.gate/up/down.weight`,
# and `block.norm_{e,h,mlp,out}.{weight,bias}`.
loop = DraftLoop.from_state_dict(sidecar_weights, lm_head_weight, depth=4, eps=1e-5)

# Eager call (returns a reused [1, depth] int64 buffer — clone if retaining across calls)
draft_ids = loop(hidden, token_ids, embed).clone()

# CUDA Graph capture & replay
loop.capture(embed)
draft_ids = loop.replay(hidden, token_ids)
```

### Verifier State Export & Rewind

Run the target model verifier with `save=True, checkpoint_interval=1` on WKV and `save_normalized=<buffer>` on TMix TokenShift and CMix. For sequence `i` starting at `start = cu_seqlens[i]` with length `L`, if `m` verified tokens are accepted:

- **`m == 0`**: Restore WKV state from `boundaries[start + i]` and restore TMix/CMix shift states from before the verifier call.
- **`0 < m < L`**: Restore WKV state from `boundaries[start + i + m]` and restore TMix/CMix shift states from `save_normalized[start + m - 1]`.
- **`m == L`**: Keep the state pools updated in place by the operators.

---

## 6. Running Tests & Benchmarks

```bash
# CPU-only contract tests
CUDA_VISIBLE_DEVICES="" ./.venv/bin/python -m pytest -q -m 'not cuda and not resource'

# Full CUDA test suite
./.venv/bin/python -m pytest -q

# WKV7 recurrent benchmark matrix (autotune with bitwise check + CUDA Graph timing)
PYTHONPATH=benchmarks ./.venv/bin/python benchmarks/tmix/wkv7/bench.py \
  --shapes h12d64 h16d64 h32d64 h40d64 h64d64 h80d64 h16d128 h8d256 \
  --cases 1x1 16x1 1x16 64x16 1x128 --tune --timing-mode graph \
  --output artifacts/wkv7-recurrent-bf16.json

# Mixed chunked serving benchmark (random slots, live metadata)
PYTHONPATH=benchmarks ./.venv/bin/python benchmarks/tmix/wkv7/bench.py \
  --shapes h64d64 --state-dtype float16 --timing-mode graph --metadata live \
  --slot-order random --state-pool-size 320 --cases 255x1+1x4 255x1+1x256 1x1024 \
  --output artifacts/wkv7-live.json

# Include per-step device metadata validation & sorting in the timed region
PYTHONPATH=benchmarks ./.venv/bin/python benchmarks/tmix/wkv7/bench.py \
  --shapes h64d64 --state-dtype float16 --timing-mode graph --metadata live \
  --include-ticket-preparation --cases 255x1+1x256
```

Every operator benchmark supports `--help` for workload flags and `--tune` (which always verifies bitwise equivalence on outputs and state before measuring latency).
