# FlashRWKV2 TileLang operator contract

模型前向统一使用 BF16 激活和权重、FP32 累加与归约。WKV 状态使用连续
`[slots,H,D,D]`，普通 serving 在 SM89、SM120 上首选 FP16，训练和 RL rollout 使用 FP32。TokenShift 状态为 BF16 `[slots,C]`。Head 输出 logits、
所选 token 的 logprob 与 loss 为 FP32。公共名称保留模块所有者；内部 primitive 不从根包导出。

## Numerical contract

训练、prefill 和 decode 使用同一份 TileLang 前向实现。投影按递增 K 顺序执行
BF16 MMA、FP32 累加，不使用 split-K；每行 ln、head ln 和词表归约使用固定顺序。
`res + x` 在 ln 前先舍入为 BF16，投影、gate、TokenShift 和 WKV 输出也在相同
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
logprob，梯度使用独立 FP32 参考与相对误差验证。WKV 反向每 16 token 使用一份
FP32 checkpoint。较长的 64 维 head 先重放每步的 `sa` 向量，再从 checkpoint
按前向算术重放短窗口内的 FP32 state。每个梯度由负责该 key 或 value 的线程直接写出。
当 batch-head 并行度较低时，长序列按 4096 token 分段，先用实际 cotangent 计算
state 梯度边界，再并行计算各段 VJP。通用路径保留 shared-memory state history 和 FP32 原子归约。
只有需要梯度时才保存 checkpoint。

## Inference

TMix 公共阶段固定为 `tokenshift -> wkv_prepare -> wkv7 -> readout`。

| API | 输入与返回值 |
|---|---|
| `infer_embedding_ln0_forward_varlen(embedding, weight, bias, *, eps=1e-5)` | BF16 embedding 和 ln 仿射参数，返回 BF16 ln0 |
| `infer_tmix_postnorm_tokenshift_forward_varlen` | res/ln、六路 mix，返回 `(res_out, r, w, k, v, a, g)`，原位更新 shift pool |
| `infer_tmix_wkv_prepare_forward_varlen` | R/K/V、低秩 W/A/G/V、VRes 与 KK/A，返回 `(r, decay_logits, key, value, recurrent_a, recurrent_b, gate, v_first)` |
| `infer_tmix_wkv7_recurrent_fp32io16_forward_varlen` | BF16 `[N,H,D]` 六路输入、FP32 state handle、packed metadata；返回 BF16 WKV 并原位更新状态 |
| `infer_tmix_wkv7_recurrent_fp16_forward_varlen` | SM89、SM120 普通 serving 首选；BF16 六路输入、FP16 state handle、FP32 算术；每 token 后将 state 舍入为 FP16 |
| `infer_tmix_readout_forward_varlen` | head ln、RKV residual、gate 和 output projection，返回 BF16；WKV 输入也可为 FP32 |
| `infer_cmix_forward_varlen` | res/ln、TokenShift、key projection、ReLU²、value projection；返回 `(res_out, cmix_out)`，原位更新 shift pool |
| `infer_post_norm_output_forward_varlen(x, res, weight, bias, *, eps=1e-5)` | 对 BF16 residual sum 作 FP32 ln，返回 BF16 |
| `infer_head_linear_all_forward_varlen(x, weight)` | `[N,C]` BF16 hidden 与 `[V,C]` BF16 weight，返回 FP32 `[N,V]` |
| `infer_sampling_six_parameter_forward_varlen` | FP32 logits、penalty pool、RNG pool 和 slot indices，返回 token ids 并更新请求状态 |

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

`prepare_tmix_wkv7_recurrent_fp32io16_state_from_tensor(state)` 绑定已有连续 CUDA
FP32 tensor。每次调用返回时状态完整写回。

`prepare_tmix_wkv7_recurrent_fp16_state_from_tensor` 绑定连续 CUDA FP16 tensor。
`infer_tmix_wkv7_recurrent_fp16_forward_varlen` 使用 BF16 token I/O、FP32 递推和归约，
每个 token 后将逻辑 state 舍入为 FP16。不同 batch、分段和 CUDA Graph 使用相同舍入规则。
对 D64、H40/H64，两个点积都在四个连续的 16-key 组内使用 FP32 FMA 顺序累加，
然后按 `(s0 + s1) + (s2 + s3)` 合并。该求和顺序同时用于单行、混合 batch 和单 token decode。

D64、H40 或 H64 的 FP16 handle 为每个 slot、每个头分配五个长度为 64 的 FP32 向量和一个
int32 phase，共 1,284 字节。批量 decode 可以把一步更新留在日志，下次读取该 slot 时按原
FP32 顺序重放并舍入。同一批次里的单 token 行独立决定追加或写回，长行优先执行。
H64 模型的 32 层每序列共用 16 MiB 基态和约 2.508 MiB 日志。
其他几何和 FP32 handle 的私有字节数为零。容量规划使用 `memory_layout["bytes_per_slot"]`，
其中包括基态和日志，`fixed_workspace_nbytes` 为零。

读取、复制或导出底层绑定 tensor 之前，对读出的槽位调用 `materialize_slots_(indices)`。
在替换、丢弃或重新绑定同一个底层 tensor 之前，同样需要物化仍要保留的槽位。
handle 的 `clone()`、`clone_slots(indices)`、`copy_()`、`copy_slots_(source, src, dst)`、
`zero_()` 和 `reset_slots_(indices)` 会处理日志。slot copy 支持同池重叠 COW，目标槽位须唯一。
训练、RL rollout、eval 和采样搜索使用 FP32 state。

静态 `prepare_tmix_wkv7_recurrent_metadata(cu_seqlens, state_indices, *,
state_pool_size, total_tokens, max_seqlen=None)` 验证序列边界、slot 范围和唯一性，
保存 snapshot，并校验后续调用的 identity、version、shape 和 CUDA stream。
`cu_seqlens` 与 `state_indices` 为连续 CUDA INT32。

CUDA Graph 使用 `token_capacity`、`sequence_capacity`、`max_seqlen_capacity` 以及
两个 CUDA INT32 scalar `num_active_tokens`、`num_active_sequences`。同一 graph
内所有 stateful 算子共享 ticket；每次 replay 验证有效前缀。零 active warmup 不
修改 state，非法 metadata 也不修改 state。inactive 输出为 NaN，不应被下游消费。
验证器在设备上按有效行长度降序排列执行顺序，结果由各层共享，输出仍按输入行排列。
FP16 prefill 按四个 token 一块处理整块，每行最后单独处理 0–3 个 token。

## Training

| API | 说明 |
|---|---|
| `pretrain_tmix_tokenshift_bf16` | 六路 TokenShift；stateful 形式返回混合输出和 final shift |
| `pretrain_tmix_wkv_prepare_bf16` | 训练权重布局与 `w0` bias，输出共享 BF16 prepare 结果 |
| `pretrain_tmix_wkv7_recurrent_bf16` | dense `[B,T,C]` 或带 cu_seqlens 的 packed `[N,H,D]`，支持任意长度和可微 FP32 初始状态 |
| `pretrain_tmix_readout_bf16` | head ln、RKV residual、gate，可选 output projection |
| `pretrain_cmix_bf16` | ChannelMix 与可微初始/最终 shift state |
| `pretrain_head_linear_bf16` | BF16 hidden/weight，返回 FP32 logits |
| `pretrain_head_ce_bf16` | `chunk_rows` 限制前向词表投影工作集，按 targets 形状返回 FP32 交叉熵，-100 对应零 loss 和梯度，`backward_chunk_rows` 可独立设置反向重算和 VJP 的行数，默认沿用 `chunk_rows` |

投影支持 LoRA/NoRA：A `[R,K]`、B `[O,R]`，参数可为 BF16/FP32。
NoRA 为 `A / (norm(A, dim=0) + 1e-6)`，FP32 归一化与反向，BF16 投影；
每次 forward 从当前参数计算。冻结 base weight 时省去 dW，但保留输入梯度。
CMix 使用 `key_*`、`value_*`，prepare 使用 `receptance_*`、`key_*`、`value_*`，
readout 使用 `output_*`，head 使用 `lora_a/lora_b/lora_scale/use_nora`。

Prepare 分别将 R/K/V、W/A/G/V 的低秩输入投影、低秩输出投影分组执行。
各组使用独立输入和权重，允许低秩宽度不同；没有 VRes 的首层只执行三路。
每路仍按原来的连续 32 元素 K 块累加，先舍入到 BF16，再执行原有 bias 与
tanh/sigmoid，最后再次舍入到 BF16。反向复用相同的激活 VJP、bias 归约和
线性 VJP；R/K/V 的 LoRA/NoRA 更新仍作用于各自的 BF16 base projection。

训练前向和反向支持 `torch.compile(fullgraph=True)`。编译路径通过注册算子调用
同一套 TileLang forward/VJP；普通 eager 调用保留原有路径。Inductor 必须保留
中间 BF16 舍入，才能与 eager 梯度逐位一致：

```python
compiled_model = torch.compile(
    model, fullgraph=True, options={"emulate_precision_casts": True},
)
```

捕获 CUDA Graph 前，用实际形状、权重布局和可训练参数预热 forward/backward，再恢复 state 和梯度。
支持 PyTorch 2.11–2.14；metadata/sampling native 扩展使用 C++20 编译。

以 `initial_state` 传入初始 WKV state。设置 `return_final_state=True` 返回
`(output, final_state)`。packed 输入使用连续 CUDA int32 `cu_seqlens`，
从 0 起单调非减，末元素为 token 总数。TokenShift 和 CMix 以
`initial_shift` 传入 shift state，同样通过 `return_final_state` 取出最终 state。

## Tuning and acceptance

### Execution configuration

RTX 4090 and PRO 6000 use deterministic projection tiles selected from the model geometry,
SM count and shared memory limit. D64 WKV selects decode or prefill from the checked-in
execution configurations. Each program retains its original FP32 reduction and state cast.
The WKV benchmark records the selected value tile, token prefetch and workload reason before capture.
The operator benchmark's `--tune` option verifies candidate outputs and state before timing.
Its report contains the winning configuration and individual measurements.

Warm the forward and, for training, the backward for each CUDA Graph shape on its capture
stream before capture. Restore the caller-owned state and gradients after warmup.
The native extension cache uses `~/.cache/flashrwkv2` or `FLASHRWKV2_CACHE_DIR`.
TileLang manages compiled binaries in its own cache.

### Validation and performance

The following measurements describe the earlier dynamic-workload implementation.

Business tests cover train/prefill/decode equivalence, independent VJPs and optimizer
updates, state ownership, frozen/unused gradients, compiled/Graph execution,
dynamic workloads and fresh-process binary-cache reuse.
Model geometry uses C=2048/2560/4096/4096 and W/A/G/V ranks
96/96/256/64, 96/96/320/64, 128/128/480/96 and 192/192/384/128.

| Measured result | SM89 | SM120 |
| --- | --- | --- |
| Business GPU suite: complete block, numerics, state, Graph and prepared workloads | 330 passed | 330 passed |
| Training workloads within 2% of the best measured comparison | 9/9 | 9/9 |
| Fresh-process disk-cache reuse | 139 cold / 0 warm device compilations | 137 cold / 0 warm device compilations |

SM89 T1024 training is 42.20 ms for 7.2B and 42.21 ms for 13.3B
(−0.39% / +0.33% versus baseline). Remaining performance target deviations are
SM89 B1 layer latency +2.41%–4.46%, 2.9B T1024 prefill combined latency +2.39%,
and SM120 1.5B B128 tuning-benefit retention of 93.77% (95% target).

Completed SM120 training acceptance (B1, layer plus Head; forward, backward and
gradient zeroing, excluding optimizer):

| Workload | Best comparison (ms) | Prepared (ms) | Latency change | Retained gain |
| --- | ---: | ---: | ---: | ---: |
| 7.2B T1024 eager | 32.786 | 32.196 | -1.80% | 119.77% |
| 13.3B T128 eager | 7.399 | 7.311 | -1.20% | 114.17% |
| 13.3B T1024 eager | 32.897 | 32.267 | -1.91% | 121.38% |
| 1.5B T128 compiled | 3.603 | 3.577 | -0.71% | 109.19% |

All four meet the 2% latency and 95% gain-retention targets. The comparison is the
fastest of baseline, existing offline reference and previous strategy. Gain retention
applies when `default - best > max(0.02 * default, 3 * (IQR(default) + IQR(best)))`.
The four cases used an idle Shanghai PRO 6000 Workstation (188 SMs), Torch
2.13.0+cu130 and CUDA 13.0; the earlier five used PRO 6000 Server. Each comparison
uses the same device/toolchain, bitwise output/state checks and ABBA timing
(7 samples × 5 Graph replays per phase). Preparation and capture are excluded.
Whole-machine monitoring found no competing GPU processes; all four preparations
reported 103/103 required variants available across 30 families.

Raw paired samples, references and numerical evidence remain in the existing
`artifacts/cmix-acceptance-a239bc3.json`,
`artifacts/sm120-remaining-acceptance-results-07c7913.json` and associated artifacts.
Performance comparisons require idle hardware, numerical checks and paired timing; unit-test
time is not performance evidence. Existing benchmarks expose implementation,
tuning and eager/Graph timing options; run `--help` for their workload arguments.
