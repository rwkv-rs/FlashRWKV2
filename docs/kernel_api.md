# FlashRWKV2 TileLang operator contract

模型前向统一使用 BF16 激活和权重、FP32 累加与归约。WKV 状态固定为连续
FP32 `[slots,H,D,D]`，TokenShift 状态为 BF16 `[slots,C]`。Head 输出 logits、
`head_logprobs` 与 loss 为 FP32。公共名称保留模块所有者；内部 primitive 不从根包导出。

## Numerical contract

训练、prefill 和 decode 使用同一份 TileLang 前向实现。投影按递增 K 顺序执行
BF16 MMA、FP32 累加，不使用 split-K；每行 LN、head LN 和词表归约使用固定顺序。
`res + x` 在 LN 前先舍入为 BF16，投影、gate、TokenShift 和 WKV 输出也在相同
位置舍入为 BF16。WKV 的 FP32 状态在 token 之间不缩窄。

WKV 输入是 raw decay logits。保留率为
`exp(-0.6065306597126334 * sigmoid(raw_decay))`。若提供 `decay_bias`，先以 FP32
相加、舍入为 BF16，再计算保留率；这与 WKV Prepare 训练入口加 `w0` 的边界相同。
状态布局为 `[K,V]`，递推为：

```text
sa[v] = sum_k a[k] * state[k,v]
state[k,v] = retention[k] * state[k,v] + b[k] * sa[v] + k[k] * v[v]
y[v] = scale * sum_k r[k] * state[k,v]
```

`head_size` 支持 64、128、256。前向一致性按位比较输出、state 和所选 token 的
logprob；梯度使用独立 FP32 参考与相对误差验证。WKV 反向从保存的 chunk 起始状态
重放前向，并在短片段内逆序求 VJP。只有需要梯度时才保存 checkpoint。

## Inference

TMix 公共阶段固定为 `tokenshift -> wkv_prepare -> wkv7 -> readout`。

| API | 输入与返回值 |
|---|---|
| `infer_embedding_ln0_forward_varlen(embedding, weight, bias, *, eps=1e-5)` | BF16 embedding 和 LN 仿射参数，返回 BF16 LN0 |
| `infer_tmix_postnorm_tokenshift_forward_varlen` | res/LN、六路 mix，返回 `(res_out, r, w, k, v, a, g)`，原位更新 shift pool |
| `infer_tmix_wkv_prepare_forward_varlen` | R/K/V、低秩 W/A/G/V、VRes 与 KK/A，返回 `(r, decay_logits, key, value, recurrent_a, recurrent_b, gate, v_first)` |
| `infer_tmix_wkv7_recurrent_fp32io16_forward_varlen` | BF16 `[N,H,D]` 六路输入、FP32 state handle、packed metadata；返回 BF16 WKV 并原位更新状态 |
| `infer_tmix_readout_forward_varlen` | head LN、RKV residual、gate 和 output projection，返回 BF16；WKV 输入也可为 FP32 |
| `infer_cmix_forward_varlen` | res/LN、TokenShift、key projection、ReLU²、value projection；返回 `(res_out, cmix_out)`，原位更新 shift pool |
| `infer_post_norm_output_forward_varlen(x, res, weight, bias, *, eps=1e-5)` | 对 BF16 residual sum 作 FP32 LN，返回 BF16 |
| `infer_head_linear_all_forward_varlen(x, weight)` | `[N,C]` BF16 hidden 与 `[V,C]` BF16 weight，返回 FP32 `[N,V]` |
| `infer_head_linear_last_forward_varlen(x, weight, *, tokens_count)` | 调用方已选出的末尾 hidden rows，返回对应 FP32 logits |
| `head_logprobs(logits, targets)` | 连续 FP32 `[...,V]` logits 和同前缀形状的 INT64 targets，返回 FP32 所选 token logprobs |
| `infer_sampling_temperature_topk_topp_forward_varlen` | FP32 logits、RNG pool、slot indices，返回 token ids 并更新 RNG |
| `infer_sampling_six_parameter_forward_varlen` | 同上，另含 presence/frequency penalties 与 penalty decay |

WKV 推理参数为：

```python
output = flashrwkv2.infer_tmix_wkv7_recurrent_fp32io16_forward_varlen(
    r, decay_logits, k, v, a, b,
    state=state,
    cu_seqlens=cu_seqlens,
    state_indices=state_indices,
    scale=1.0,
    decay_bias=None,
    max_seqlen=None,
    validated_metadata=ticket,
)
```

原始低秩 checkpoint 权重按 `[out,in]` 传入推理 prepare；可选 `*_runtime`
权重是它们的转置布局。两者通过 stride 进入相同投影，不在每次调用时复制权重。
CMix 推理的 value weight 保持 `[F,C]` runtime 布局；训练入口为 `[C,F]`。

## State and packed metadata

`prepare_tmix_wkv7_recurrent_fp32io16_state(state_pool_size, channels, *,
sequence_capacity, head_size=64, device=None)` 分配零初始化 FP32 pool。
`prepare_tmix_wkv7_recurrent_fp32io16_state_from_tensor(state)` 绑定已有连续 CUDA
FP32 tensor。每次调用返回时状态完整写回。

handle 提供 `clone()`、`copy_()`、`zero_()`、`clone_slots(indices)`、
`copy_slots_(source, source_indices, destination_indices)` 和 `reset_slots_(indices)`。
slot copy 具有同时复制语义，支持同池重叠 COW；目标 slots 必须唯一。
`materialize_slots_` 保留 lifecycle 调用入口，FP32 状态本身已物化。
`memory_layout` 报告实际 bytes per slot 和 total bytes。

静态 `prepare_tmix_wkv7_recurrent_metadata(cu_seqlens, state_indices, *,
state_pool_size, total_tokens, max_seqlen=None)` 验证非空序列、边界、slot 范围和唯一性，
保存 snapshot，并校验后续调用的 identity、version、shape 和 CUDA stream。
`cu_seqlens` 与 `state_indices` 为连续 CUDA INT32。

CUDA Graph 使用 `token_capacity`、`sequence_capacity`、`max_seqlen_capacity` 以及
两个 CUDA INT32 scalar `num_active_tokens`、`num_active_sequences`。同一 graph
内所有 stateful 算子共享 ticket；每次 replay 验证有效前缀。零 active warmup 不
修改 state，非法 metadata 也不修改 state。inactive 输出为 NaN，不应被下游消费。

## Training

| API | 说明 |
|---|---|
| `pretrain_tmix_tokenshift_bf16` / `statetune_tmix_tokenshift_bf16` | 六路 TokenShift；stateful 形式返回混合输出和 final shift |
| `pretrain_tmix_wkv_prepare_bf16` | 训练权重布局与 `w0` bias，输出共享 BF16 prepare 结果 |
| `pretrain_tmix_wkv7_recurrent_bf16` | `[B,T,C]`、零初始状态，返回同形状 BF16 输出 |
| `statetune_tmix_wkv7_recurrent_fp32io16` | 指定 segment 和初始 FP32 state，返回 `(output, final_state)` |
| `rl_infctx_tmix_wkv7_recurrent_bf16` | packed 六路输入、`state_pool`、`cu_seqlens`、可选 `state_indices`，返回 `(output, final_pool)`；输入和 state 可求梯度 |
| `pretrain_tmix_readout_bf16` | head LN、RKV residual、gate，可选 output projection |
| `pretrain_cmix_bf16` / `statetune_cmix_bf16` | ChannelMix 与可微初始/最终 shift state |
| `pretrain_tmix_a_gate_bf16` / `pretrain_tmix_kk_pre_bf16` / `pretrain_tmix_vres_gate_bf16` | prepare 组成部分，复用相同前向 |
| `pretrain_head_linear_bf16` | BF16 hidden/weight，返回 FP32 logits |
| `pretrain_l2wrap_ce_bf16` | FP32 CE，梯度包含 train_temp L2Wrap 项 |
| `pretrain_head_l2wrap_ce_bf16` | `chunk_rows` 限制词表投影工作集，反向重算当前 chunk |

投影支持 LoRA/NoRA：A `[R,K]`、B `[O,R]`，参数可为 BF16/FP32。
NoRA 为 `A / (norm(A, dim=0) + 1e-6)`，FP32 归一化与反向，BF16 投影；
每次 forward 从当前参数计算。冻结 base weight 时省去 dW，但保留输入梯度。
CMix 使用 `key_*`、`value_*`，prepare 使用 `receptance_*`、`key_*`、`value_*`，
readout 使用 `output_*`，head 使用 `lora_a/lora_b/lora_scale/use_nora`。

## Tuning and acceptance

内部 `tune_linear`、`tune_norm`、`tune_tokenshift`、`tune_readout` 与 WKV tuner
按设备及实际 shape 选择配置。先与固定数值基准逐位比较，再用 CUDA Graph 计时；
候选只能改变输出 tiling、pipeline 或每 CTA 的独立行数。验收覆盖 RTX 4090
SM89 与 PRO6000 SM120 的 decode、小 batch、prefill 和训练形状。

`tests/tmix/wkv7/test_block.py` 覆盖完整 block 到 selected logprobs 的训练/推理、
prefill/decode、batch 和 optimizer 更新后一致性。`tests/tmix/wkv7/test.py` 覆盖
state lifecycle 与 CUDA Graph。设备实测通过前，离线 lowering 不能作为数值或性能
验收结果。下游模型与 serving 集成应在各自仓库单独接入这一契约。
