# FlashRWKV2 TileLang operator contract

模型前向统一使用 BF16 激活和权重、FP32 累加与归约。默认 WKV 状态为连续
FP32 `[slots,H,D,D]`，TokenShift 状态为 BF16 `[slots,C]`。Head 输出 logits、
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
| `infer_tmix_wkv7_recurrent_fp16_forward_varlen` | 可选低精度推理；BF16 六路输入、FP16 state handle、FP32 算术；每 token 后将 state 舍入为 FP16 |
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

`prepare_tmix_wkv7_recurrent_fp16_state_from_tensor` 绑定已有连续 CUDA FP16 tensor，
但 pool 使用 FP16，显存占用为 FP32 的一半。它们只供
`infer_tmix_wkv7_recurrent_fp16_forward_varlen` 使用，token 输入和输出仍为 BF16，
递推算术和归约仍为 FP32；每个 token 更新后将 state 舍入为 FP16。
该可选路径在不同 batch、prefill 分段、decode 和 CUDA Graph 间保持自身的逐位一致性，
不承诺与 FP32 训练轨迹逐位一致。训练、RL/Infctx 和 state tuning 继续要求 FP32 state。

handle 提供 `clone()`、`copy_()`、`zero_()`、`clone_slots(indices)`、
`copy_slots_(source, source_indices, destination_indices)` 和 `reset_slots_(indices)`。
slot copy 具有同时复制语义，支持同池重叠 COW；目标 slots 必须唯一。
`materialize_slots_` 保留 lifecycle 调用入口，FP32 状态本身已物化。
`memory_layout` 报告实际 bytes per slot 和 total bytes。

静态 `prepare_tmix_wkv7_recurrent_metadata(cu_seqlens, state_indices, *,
state_pool_size, total_tokens, max_seqlen=None)` 验证序列边界、slot 范围和唯一性，
保存 snapshot，并校验后续调用的 identity、version、shape 和 CUDA stream。
`cu_seqlens` 与 `state_indices` 为连续 CUDA INT32。

CUDA Graph 使用 `token_capacity`、`sequence_capacity`、`max_seqlen_capacity` 以及
两个 CUDA INT32 scalar `num_active_tokens`、`num_active_sequences`。同一 graph
内所有 stateful 算子共享 ticket；每次 replay 验证有效前缀。零 active warmup 不
修改 state，非法 metadata 也不修改 state。inactive 输出为 NaN，不应被下游消费。

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

先用 eager 的完整 forward/backward 函数执行 `prepare_tuning`，再编译、预热和
捕获。训练参数集合、LoRA/NoRA 模式与实际训练一致；仅 forward 不能覆盖 dX、dW
等反向投影 key。准备函数负责清理梯度并恢复可变 state；调优完成后再开始参数更新。
支持 PyTorch 2.11–2.14；metadata/sampling native 扩展使用 C++20 编译。

以 `initial_state` 传入初始 WKV state。设置 `return_final_state=True` 返回
`(output, final_state)`。packed 输入使用连续 CUDA int32 `cu_seqlens`，
从 0 起单调非减，末元素为 token 总数。TokenShift 和 CMix 以
`initial_shift` 传入 shift state，同样通过 `return_final_state` 取出最终 state。

## Tuning and acceptance

### Execution and preparation

Validated deterministic strategies cover RTX 4090 (SM89, 128 SMs) and
PRO 6000 (SM120, 188 SMs), BF16 projections with BF16/FP32 outputs, and
head-size-64 WKV with FP32 state. Other devices, mixed grouped layouts,
FP16 state and other head sizes retain verified profiles and bounded tuning.
The numerical contract above applies to every execution mode.

Single and grouped projections share a resource-aware tile selector based on
output geometry, actual SM count, CTA coverage and shared-memory limits.
BF16 weight gradients use dynamic token reduction: aligned K uses a two-stage
pipeline, tails use synchronous single-stage copies, and both retain consecutive
32-element MMA groups. TokenShift uses dynamic batch and 256-column blocks.
WKV selects one-token prefetch for decode and four-token prefetch for multi-token
workloads. Norm and Readout preserve their existing statistics/residual semantics.

```python
from flashrwkv2 import prepare_tuning

# Real calls with disposable/reset state; include backward when training.
report = prepare_tuning(warmup_calls, mode="require")
assert report["complete"]
# Restore serving state, then specialize torch.compile and capture CUDA Graphs.
```

Callers declare all actual shape families: static model dimensions, layouts,
dtypes, grouped branches, bias/activation forms, output modes and gradient needs.
Include first/later layers, decode/prefill and training backward. Rows, token counts
and same-shape weight addresses do not define additional deterministic families.
New layouts or trainable-parameter/output-use patterns require another preparation.
Empty calls do not prepare future nonempty workloads.

Preparation compiles every reachable variant of the observed deterministic
families. Afterwards, changing batch/token within those families performs no
TileLang code generation, device compilation, configuration search or timing.
Normal eager calls compile only their selected variant. torch.compile specialization
and CUDA Graph capture shapes are separate initialization responsibilities.

| Family | Maximum variants per static geometry/semantic form |
| --- | ---: |
| Static-K projection | 5 |
| Dynamic-K projection, fixed output geometry | 2 alignment classes |
| PostNorm / TokenShift forward | 1 |
| PostNorm / TokenShift / Readout backward, per gradient requirement | 6 |
| Readout / WKV forward, head size 64 and FP32 WKV state | 2 |
| WKV backward, head size 64 without status masking | 5 |
| Other WKV backward; state transfer per gather/scatter direction | 1 |

Head-size-64 backward prepares shared replay, direct checkpoint replay and the
segment-boundary kernels for the observed bias and scaling form. A short
training warmup covers the packed-token and batch-head dispatch thresholds.

Single-variant auxiliary kernels are warmed by the real calls. A required compile
failure aborts preparation with its owner and key; it is never silently skipped.
`compilation` reports `families`, `required_variants`, `available_variants` and
`complete`. Top-level `complete` also requires no missing observed configurations.

### Profiles, caches and graphs

`inspect` reports coverage without tuning; `require` fails on the first missing
profile; `tune` verifies then measures missing profiles. Deterministic paths report
`fixed` and bypass profiles/search in all modes. Other statuses are `hit`, `missing`
and `tuned`; records contain owner, device/key, configuration, fingerprint and path.

The cache root is `~/.cache/flashrwkv2`, overridden only by `FLASHRWKV2_CACHE_DIR`
set before import. Verified profiles live under `tuning/v2/<owner>/<fingerprint>/`;
source AST/dependencies, toolchain versions and compile flags determine validity.
Installation paths, comments and Torch/XDG cache locations do not invalidate them.
Native extensions and TileLang binaries have separate compatibility/cache checks.

Concurrent processes lock and recheck a missing profile before atomically publishing.
Eager calls recheck missing files; explicit preparation reloads each observed key.
Corrupt profiles fail validation. Preparation is exclusive within a process and
includes autograd worker dispatches; finish serving/training work before preparing.
Legacy profiles are not migrated: rerun preparation after upgrading. TileLang's
own compatible compilation cache remains reusable.

Load profiles and compile before capture. Default-kernel warmup also supports
capture; publishing a new configuration requires loading and warming it outside
capture, then recapturing existing graphs. Same-shape weight replacement reuses
compiled variants; update captured weights with `copy_` on the original storage.
Tuning samples must contain valid, nonempty active metadata; zero-active graph
warmup is supported separately and is not a tuning sample.

### Validation and performance

CPU runtime/profile checks: 67 passed. SM89/SM120 device-code regressions:
12 passed.

Business tests cover train/prefill/decode equivalence, independent VJPs and optimizer
updates, state ownership, frozen/unused gradients, compiled/Graph execution,
prepared dynamic workloads and real fresh-process profile/binary-cache reuse.
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
