# FlashRWKV2 0.1.0a10 operator contract

推理公共接口只包含模型调用者需要的最高融合岛。内部 launcher 不属于 Python 或 native 公共契约，也不会从根包导出。

## Inference

下表是 `transformers-rwkv` 需要适配的完整推理契约。省略的类型均为
`torch.Tensor`；`ticket` 表示 `prepare_tmix_wkv7_recurrent_metadata` 的返回值。

| API | 参数 | 返回值 |
|---|---|---|
| `infer_embedding_ln0_forward_varlen` | `embedding, weight, bias, *, eps=1e-5` | embedding + LN0 结果 |
| `infer_tmix_postnorm_tokenshift_forward_varlen` | `x, res, weight, bias, x_r, x_w, x_k, x_v, x_a, x_g, *, shift_state_pool, cu_seqlens, state_indices, max_seqlen=None, eps=1e-5, validated_metadata=None` | `(res_out, r, w, k, v, a, g)`；原位更新 shift state |
| `infer_tmix_wkv_prepare_forward_varlen` | `x_r, x_w, x_k, x_v, x_a, x_g, receptance_weight, key_weight, value_weight, w1, a1, g1, v1, w2, a2, g2, v2, v0, k_k, a0, k_a, *, v_first=None, w1_runtime=None, a1_runtime=None, g1_runtime=None, v1_runtime=None, w2_runtime=None, a2_runtime=None, g2_runtime=None, v2_runtime=None, receptance_lora_a=None, receptance_lora_b=None, receptance_lora_scale=1.0, key_lora_a=None, key_lora_b=None, key_lora_scale=1.0, value_lora_a=None, value_lora_b=None, value_lora_scale=1.0, head_size=64, batch_size=1, max_seqlen=None` | 一次 API 完成 R/K/V、WAG/WAGV、VRes 与 KK/A，返回 `(receptance, decay_delta, key, value, recurrent_a, recurrent_b, gate, v_first)` |
| `infer_tmix_readout_forward_varlen` | `wkv_output, receptance, key, value, r_k, ln_weight, ln_bias, gate, output_weight, *, output_lora_a=None, output_lora_b=None, output_lora_scale=1.0, head_size=64, batch_size=1, max_seqlen=None` | 完成 head LN、RKV residual、gate 和 output projection |
| `infer_tmix_wkv7_recurrent_fp32io16_forward_varlen` | `r, decay_logits, k, v, a, b, *, state, cu_seqlens, state_indices, scale=1.0, decay_bias=None, max_seqlen=None, validated_metadata=None` | WKV 输出；原位更新统一 FP32IO16 state handle，并在内部选择普通 FP32IO16 或 DeltaLog |
| `infer_tmix_wkv7_recurrent_fp16_forward_varlen` | `r, decay_logits, k, v, a, b, *, state, cu_seqlens, state_indices, scale=1.0, decay_bias=None, max_seqlen=None, validated_metadata=None` | WKV 输出；原位更新统一 FP16 state handle，并在内部选择普通 FP16 或 DeltaLog |
| `infer_tmix_wkv7_chunk_bf16_forward_varlen` | `r, decay_logits, k, v, a, b, *, state_pool, cu_seqlens, state_indices, chunk_size=16, max_seqlen=None, scale=1.0, decay_bias=None, validated_metadata=None` | chunk WKV 输出；原位更新 state |
| `infer_cmix_forward_varlen` | `x, res, weight, bias, x_k, key_weight, value_weight, *, shift_state_pool, cu_seqlens, state_indices, max_seqlen=None, eps=1e-5, validated_metadata=None, deterministic=False` | `(res_out, cmix_out)`；原位更新 shift state |
| `infer_post_norm_output_forward_varlen` | `x, res, weight, bias, *, eps=1e-5` | 最终 `LN(x + res)` |
| `infer_head_linear_all_forward_varlen` | `x, weight` | 所有 packed token 的 logits |
| `infer_head_linear_last_forward_varlen` | `x, weight, *, tokens_count` | 最后 `tokens_count` 个 token 的 logits |
| `infer_sampling_temperature_topk_topp_forward_varlen` | `logits, states, slot_indices, *, temperature=1.0, top_k=-1, top_p=1.0, sample_capacity=None, num_active_samples=None` | sampled token ids；原位更新 RNG state |
| `infer_sampling_six_parameter_forward_varlen` | `logits, penalties, states, slot_indices, *, presence_penalty=0.0, frequency_penalty=0.0, penalty_decay=0.996, temperature=1.0, top_k=-1, top_p=1.0, sample_capacity=None, num_active_samples=None` | sampled token ids；原位更新 penalty/RNG state |

`prepare_tmix_wkv7_recurrent_fp16_state(state_pool, elapsed_state_pool, *,
sequence_capacity)` 与 `prepare_tmix_wkv7_recurrent_fp32io16_state(state_pool,
*, sequence_capacity)` 分别返回不透明的统一 FP16 与 FP32IO16 state handle。
FP16 handle 持有调用方提供的 FP16 基础 state 与 elapsed state；FP32IO16 handle
持有 FP32 基础 state。两者在需要时都一次性分配与 state 同 dtype 的私有
DeltaLog phase/log workspace；`clone()`、`copy_()` 和 `zero_()` 始终覆盖完整逻辑
状态包。下游不得读取、移动或单独管理 DeltaLog policy、`M`、phase 或 logs。

策略固定自 Albatross `3465da5070beceb4bab9e07b03abee1642a0bdf8` 的普通
`WKV_DELTALOG_TUNED_M` 精确表。只有 SM120、FP16 token IO、FP16 或 FP32
state、`D=64`、固定容量 `T=1` 且 `(C, sequence_capacity)` 精确命中该表时，
对应统一入口才调用私有 DeltaLog launcher；FP32 state 的 BF16 IO、`T>1`、其他
device/head size 或表外 shape 自动调用对应普通 launcher。策略不插值，不接受
外部 `M` override，也不包含依赖模型级 CUDA Graph 调度的 APW-only 表。策略与
workspace 在 state preparation 阶段确定；统一入口不会在热路径查询主机侧 CUDA
数据。native 校验异常继续令整套 state fail-closed。

`prepare_tmix_wkv7_recurrent_fp16_state`、
`prepare_tmix_wkv7_recurrent_fp32io16_state`、
`prepare_tmix_wkv7_recurrent_metadata` 和 `setup_sampling_states` 是上述推理入口
需要的状态构造 API。

训练接口保持 14 个既有完整语义入口：`pretrain_cmix_bf16`、
`statetune_cmix_bf16`、`pretrain_tmix_wkv7_recurrent_bf16`、
`statetune_tmix_wkv7_recurrent_fp32io16`、`rl_infctx_tmix_wkv7_chunk_fp32io16`、
`rl_infctx_tmix_wkv7_chunk_fp32io16_factor_recompute`、`pretrain_tmix_a_gate_bf16`、
`pretrain_tmix_kk_pre_bf16`、`pretrain_tmix_readout_bf16`、
`pretrain_tmix_tokenshift_bf16`、`statetune_tmix_tokenshift_bf16`、
`pretrain_tmix_vres_gate_bf16`、`pretrain_l2wrap_ce_bf16` 和
`pretrain_head_l2wrap_ce_bf16`。根包总计导出 31 个唯一名称：13 个推理入口、
14 个训练入口和上述 4 个状态构造入口。

TMix 的 `B=1,T=1,C=4096` 与 CMix 的 `T=1,C=4096` 使用 Albatross 派生的单次 Res+LN+TokenShift fused launch；其他 packed varlen 形状在同一公共入口内部顺序启动 PostNorm 与 TokenShift。CMix 的 dense/sparse 选择也完全由 `infer_cmix_forward_varlen` 内部完成。

## Downstream handoff

`transformers-rwkv` 必须精确依赖 `FlashRWKV2==0.1.0a10`，只检查和调用上述公共
接口。模型侧不得为 FP16 或 FP32IO16 选择普通/DeltaLog、传入 `M`、管理
phase/logs，或调用
standalone LN、Res、TokenShift、CMix ReLU²、FFN-down、sparse 子步骤、WKV
elapsed-state 更新及通用 Linear helper。
