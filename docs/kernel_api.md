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

投影先以 32 元素的 K 拷贝跨度选择输出 tile，再为获胜 tile 测量 64/128 元素的跨度。
同时测量 M 取输入行数向上取整到 2 的幂（限于 16–256）的最佳 tile，最多增加
四个大拷贝候选。较大的跨度必须整除输入宽度。每次拷贝内部仍按连续 32 元素块执行相同的 MMA 序列，
保持 MMA 次数与 FP32 累加顺序。未显式调优时，默认 M tile 按行数从
16/32/64/128 中选择；只有 CTA 网格可覆盖设备 SM 时才增大 M tile。
超过 128 行的投影会测量 256 行 tile，较大的 tile 使用最多 256 个线程复用权重，
MMA 的 K 序列保持不变。

调优同时核对 FP32 累加或统计量及实际生产输出；LN 是否保存统计量、是否写回
res 等变体都必须保持相同结果。线性层的 bias presence/dtype 属于调优 key。
编译不受工具链支持的线性候选会在结果中记录 `compile_error`；运行失败直接报错。
候选的 staging buffers 超出设备共享内存上限时，记录 `resource_error` 并在启动前排除。
调优需在 CUDA Graph capture 前完成，并使用有效、非空的 active workload。
WKV 的 functional 与 indexed pool 模式分别缓存配置；候选同时校验两种模式的
输出和 FP32 最终状态，按实际调用模式计时。

PostNorm、TMix 各推理阶段、CMix 和 Head 线性基准可用
`--implementation tilelang|cuda-baseline` 标记
通过 `PYTHONPATH` 选中的包。两次运行复用同一基准脚本、seed、逻辑输入及尺寸，
结果记录实际输入、输出和上游梯度 dtype。WKV 基线使用 owning FP32IO16 handle，
保留原 CUDA selector 的生产路径；metadata preparation、state reset 和 materialize
位于计时外。`--timing-mode eager` 测完整 Python dispatch 与末尾同步的 wall time，
`--timing-mode graph` 用 CUDA events 测包含整批调用的 graph replay；两种结果分别比较。
Graph replay 与 events 在正式计时前初始化，额外的整批 Graph 预热由
`graph_warmup_launches` 单独记录；默认正式计时仍为 10 批、每批 1000 次调用。
Head CE 基准的 `--backward`、`--train-base`、`--chunk-rows` 明确反向范围及工作集，
并报告单次调用的 peak allocated memory。

Head 线性基准的 `--compare-torch` 用相同 BF16 输入，对照输出 FP32 的原生
`torch.mm`。对照时关闭 BF16 reduced-precision reduction 与 split-K，记录原生调用在整批、
单行和七行分块之间的 FP32/BF16 逐位一致性及独立计时，以验证 custom linear
是否有必要。原生库满足一致性和性能要求时，应优先减少自定义实现。
样本包含 BF16 有效位和不同指数；单行输入还会扩展到 17 行，核对不同矩阵行数下
同一行的结果。

`--compare-torch` 在首次 CUDA 调用前选择 cuBLASLt，并设置
`CUBLAS_WORKSPACE_CONFIG=:16:8` 和
`CUBLASLT_WORKSPACE_SIZE=1`；artifact 记录实际 preferred BLAS 与 workspace 设置。
该配置参考 vLLM 的 batch-invariant 初始化，仍须验证本项目 BF16/FP32 输出契约。
CI 的 `tuned_multisize` workload 在六种模型宽度和 M=1/27/256 共 18 种形状下，
同时记录 TileLang 调优结果与 cuBLASLt 原生对照。

CI 基准结果同时保留每个 workload 的完整 JSON records 和延迟 profiles。JSONL
尺寸 sweep 的全部记录、调优配置、原生库对照与显存数据均进入上传的基准 artifact。

`tests/tmix/wkv7/test_block.py` 覆盖完整 block 到 selected logprobs 的训练/推理、
prefill/decode、batch 和 optimizer 更新后一致性。`tests/tmix/wkv7/test.py` 覆盖
state lifecycle 与 CUDA Graph。设备实测通过前，离线 lowering 不能作为数值或性能
验收结果。下游模型与 serving 集成应在各自仓库单独接入这一契约。

## 上游实现与实现边界

2026-10-20 核对 vLLM
[`005cef040a28`](https://github.com/vllm-project/vllm/tree/005cef040a283aac2b156eccb12a2ab5d10f3157)：

- 普通非量化线性层默认调用 `torch.nn.functional.linear`；开启 batch invariance
  后改用独立 Triton GEMM。其配置按架构和 `(N,K)` 选择，再按 M 分桶，保持同一
  shape 的 `BLOCK_K` 不随 batch 改变。见
  [linear](https://github.com/vllm-project/vllm/blob/005cef040a283aac2b156eccb12a2ab5d10f3157/vllm/model_executor/layers/linear.py) 与
  [batch-invariant configs](https://github.com/vllm-project/vllm/blob/005cef040a283aac2b156eccb12a2ab5d10f3157/vllm/model_executor/determinism/batch_invariant_configs.py)。
- Qwen3.5 的 GDN prefill 组合 FLA/Triton、FlashInfer 和可选的 SM100 CuTe DSL；
  SM120 通过 FlashInfer 的对应实现支持。decode 另有面向 SM80+ 的 fused CUDA。
  见 [Qwen GDN](https://github.com/vllm-project/vllm/blob/005cef040a283aac2b156eccb12a2ab5d10f3157/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py)。
- Kimi-K3 的 KDA prefill 使用 FlashKDA CUDA/CUTLASS 或 Triton；decode 有
  fused CUDA 及 Triton。低延迟 BF16 GEMM 另按设备、shape 和 M 保存实测配置，
  部分路径在 batch-invariant 模式下禁用。见
  [KDA](https://github.com/vllm-project/vllm/blob/005cef040a283aac2b156eccb12a2ab5d10f3157/vllm/models/kimi_k3/nvidia/kda.py) 与
  [low-latency GEMM](https://github.com/vllm-project/vllm/blob/005cef040a283aac2b156eccb12a2ab5d10f3157/vllm/models/kimi_k3/nvidia/low_latency_gemm.py)。
  可选 `decode_backend=latency_specialized` 进一步使用 CuTe DSL executor，入口
  限定 SM100/103、TP8/16 等部署条件，见
  [specialized decode](https://github.com/vllm-project/vllm/blob/005cef040a283aac2b156eccb12a2ab5d10f3157/vllm/models/kimi_k3/nvidia/specialized_decode/runtime/__init__.py)。

本项目的实现选择以数值契约、实测性能及维护成本决定。普通 GEMM 优先验证原生
BLAS；自定义实现须由 batch/prefill/decode 一致性或实际融合收益支持。DSL 不消除
shape 调优；仅保留有测量依据的执行配置。SM100 与 SM120 分别验证，不按 Blackwell
名称推定指令和资源限制相同。KDA/GDN 与 RWKV7 的递推公式不同，复用应核对数学
和接口契约。
