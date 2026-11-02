# FlashRWKV2 TileLang operator contract

模型前向统一使用 BF16 激活和权重、FP32 累加与归约。WKV 状态固定为连续
FP32 `[slots,H,D,D]`，TokenShift 状态为 BF16 `[slots,C]`。Head 输出 logits、
`head_logprobs` 与 loss 为 FP32。公共名称保留模块所有者；内部 primitive 不从根包导出。

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
logprob；梯度使用独立 FP32 参考与相对误差验证。WKV 反向从保存的 chunk 起始状态
重放前向，并在短片段内逆序求 VJP；不同 value tile 的部分梯度通过 FP32 原子加累积，
调度顺序会造成极小的舍入差异。前向输出和 state 逐位比较，反向沿用独立 VJP 测试的
`atol=2e-5, rtol=2e-2` 容差。对 200 组固定输入和上游梯度的复测中，
相同 `(64,1)` 配置重复运行有 4 组梯度不完全相同，`(64,1)` 与 `(64,4)`
比较有 5 组不同；最大绝对差为 3.05e-5，前向输出和 state 均逐位一致。
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
| `infer_head_linear_last_forward_varlen(x, weight, *, tokens_count)` | 调用方已选出的末尾 hidden rows，返回对应 FP32 logits |
| `head_logprobs(logits, targets)` | 连续 FP32 `[...,V]` logits 和同前缀形状的 INT64 targets，返回 FP32 所选 token logprobs |
| `infer_sampling_temperature_topk_topp_forward_varlen` | FP32 logits、RNG pool、slot indices，返回 token ids；零温度确定性 argmax 不更新 RNG，正温度抽样更新 RNG |
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

`prepare_tmix_wkv7_recurrent_fp16_state` 和
`prepare_tmix_wkv7_recurrent_fp16_state_from_tensor` 提供相同的容量参数及 lifecycle，
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
| `pretrain_tmix_readout_bf16` | head ln、RKV residual、gate，可选 output projection |
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

StateTune 在编译或 capture 前验证 chunk metadata：

```python
from flashrwkv2 import (
    prepare_tmix_wkv7_statetune_metadata,
    statetune_tmix_wkv7_recurrent_fp32io16,
)

metadata = prepare_tmix_wkv7_statetune_metadata(
    sequence_chunk_offsets, chunk_token_starts, chunk_token_ends,
    total_tokens=r.shape[0],
)
output, final_state = statetune_tmix_wkv7_recurrent_fp32io16(
    initial_state, sequence_chunk_offsets, chunk_token_starts, chunk_token_ends,
    r, decay_logits, k, v, a, b, prepared_metadata=metadata,
)
```

返回的 metadata record 持有独立 CUDA INT32 `cu_seqlens` 和 `total_tokens`。
使用期间保持 offsets 不变；序列分段或 token 总数变化时重新准备并重新捕获。
传入 `prepared_metadata` 后，执行路径使用该 record，不再读取原始 chunk 数组。
未传入时，eager 调用仍在每次执行前验证并转换原有 chunk 参数。

## Tuning and acceptance

### 确定性执行策略

确定性路径面向 SM89/128 SM 的 RTX 4090 与 SM120/188 SM 的 PRO 6000，
使用 BF16 输入、BF16/FP32 投影输出及 FP32 WKV state。普通前向不扫描候选，
确定性分派在 `inspect`、`require` 和 `tune` 中均不读取 profile、不搜索或计时。
其他设备、混合投影布局和显式 FP16 state 等路径保留既有 profile 语义。

普通矩阵投影与 WKV 分组投影共用输出 tile 选择器。输入为实际行数、各路输出
宽度、归约宽度或动态 K 标记，以及设备的 SM 数量与共享内存预算；不使用模型
名称或模型专属行数阈值。菜单如下：

| 工作量 | 输出 tile 菜单 |
| --- | --- |
| 不超过 16 行 | M=16，N 从 128/64/32/16 中按 CTA 覆盖选择 |
| 每路输出不超过四个 128 列块 | (64,64)、(32,32)、(16,32)、(16,16) |
| 静态 K，至少一路输出的 128 列块数达到 SM 数 | (256,64)、(128,64)、(64,64)、(32,64) |
| 其他投影 | (128,128)、(64,128)、(64,64)、(32,64) |

分组投影将每个输出 tile 的累加器面积限制为 8192 元素，宽列矩阵通过较大行
tile 复用权重。策略在候选中优先选择满足一轮 SM 覆盖的较大 tile，并限制行尾浪费。
CTA 数为 `ceil(rows/M) * sum(ceil(outputs_i/N))`。拷贝跨度受静态 K 整除、
共享内存和流水约束限制，每次仍执行相同的连续 32-K MMA 组。CTA 覆盖是
并行度启发式；缓存、寄存器与访存成本通过整层配对验证。

分组反向按计算结构、布局、梯度需求和输出使用情况决定能否合并，与是否存在
固定配置表解耦。输入 VJP 与权重 VJP 分别判断；冻结参数和未使用输出保持
逐路语义。单投影和分组 BF16 权重 VJP 的 token 归约维为动态 K：对齐路径采用
双级流水及 32 元素拷贝，其他 K 采用单级流水、32 元素拷贝和同步加载。
对齐动态 K 将 `K % 32 == 0` 显式传入编译器，保留动态行跨度的向量异步加载；
拷贝跨度与单个 MMA 组相同，避免部分拷贝阶段的谓词限制。动态 K 宽输出使用
常规菜单，不采用静态宽列投影的 256 行 tile。
短归约和尾部加载补零并保持递增的 32-K 归约顺序。FP32 上游梯度继续使用既有
原生矩阵乘路径。编译算子的分组梯度输出保留必要的独立存储复制。

PostNorm、TokenShift 和 Readout 保留各自的归约与融合语义；TokenShift 使用
固定的 256 列块。WKV 在 head size 64、FP32 state 下，decode 使用单 token
预取，多 token 使用四 token 预取，逐 token state 更新顺序不变。

### 显式准备与有限编译变体

普通调用仅按需编译当前配置。`prepare_tuning` 对实际观察到的确定性形状族
枚举并准备其全部可达配置：除起始行数外，还检查输出 tile 生效边界与 CTA
条件首次满足的位置。给定列块数 S，`ceil(rows/M)*S >= SM_count` 的首次
满足位置为 `(ceil(SM_count/S)-1)*M+1`。动态 K 准备有限的对齐类别；
Norm、Readout 和 TokenShift 的反向行分组也在准备阶段覆盖。

每个静态投影形状族最多有 5 个变体：小行数菜单只选出一个配置，其余行数
最多从四个输出 tile 中选择；K 拷贝跨度由该 tile、静态 K 和设备预算唯一确定。
动态 K 的输出尺寸固定，每个形状族最多 2 个变体，分别覆盖对齐和尾部类别。
其他可达变体的上界为：

| 形状族 | 必需变体上界 |
| --- | ---: |
| PostNorm / TokenShift 前向的每种融合语义 | 1 |
| PostNorm / TokenShift 反向的每种梯度需求 | 6 |
| Readout 前向（head size 64） | 2 |
| Readout 反向的每种梯度需求 | 6 |
| WKV 前向（head size 64、FP32 state） | 2 |
| WKV 反向 | 1 |
| 状态池 gather / scatter 的每个方向 | 1 |

单变体辅助 kernel 由真实准备调用直接预热；报告统计经有限变体准备入口登记的形状族。

形状族由 owner、架构、静态尺寸、布局、dtype、分组路数、激活/bias 和输出
模式等代码生成因素定义，不含动态 batch/token、权重地址或模型路径。
调用方应提供实际使用的首层/后续层、decode/prefill 和梯度需求的真实调用。
新增布局、冻结参数模式或输出使用方式时，将相应调用纳入初始化准备。

报告保留现有字段，并新增 `compilation`：

- `families`：本次实际观察并准备的编译形状族数量。
- `required_variants` / `available_variants`：这些形状族的必需/可用变体数。
- `complete`：必需变体均已成功准备；失败会报告 owner 与编译键并中止。

顶层 `complete` 同时要求观察到的配置没有缺失。报告描述调用集合的实际覆盖，
调用方负责使其覆盖预期使用范围。准备后，在这些确定性形状族内改变 batch/token
不会再触发 TileLang 代码生成、设备编译、配置搜索或性能计时。
`torch.compile` 的动态维度设置和 CUDA Graph 捕获形状由调用方在初始化时确定。
同规格权重替换复用执行配置与编译缓存；Graph 中通过 `copy_` 写回原权重存储。

### 验收与测量

双架构数值与准备后零编译验收通过。本次动态归约修正已在 SM89 完成训练矩阵，
SM120 完成控制变量实验及五项训练配对；其余训练和新进程缓存复测等待空闲整机。
未修改的推理路径仍有性能目标偏差，PR #24 保持 draft。
验证设备为节点 125 的 RTX 4090（SM89、128 SM）和节点 106 的 RTX PRO 6000
Blackwell Server Edition（SM120、188 SM），均使用 PyTorch 2.14.1+cu130。
四组模型尺寸采用真实宽度及 W/A/G/V rank：

| 模型 | C | W/A/G/V rank |
| --- | ---: | --- |
| 1.5B | 2048 | 96 / 96 / 256 / 64 |
| 2.9B | 2560 | 96 / 96 / 320 / 64 |
| 7.2B | 4096 | 128 / 128 / 480 / 96 |
| 13.3B | 4096 | 192 / 192 / 384 / 128 |

| 检查 | SM89 | SM120 |
| --- | --- | --- |
| 既有完整基线的 CUDA 数值、梯度、Graph 与准备行为用例 | 62 项通过 | 62 项通过 |
| 本次动态归约相关集中回归（含编译代码检查） | 28 项通过 | 28 项通过 |
| 完整 block、两层状态、独立 VJP / optimizer 回归 | 通过 | 通过 |
| 准备后的新 batch/token、菜单边界及固定种子随机样本 | 0 次新增 TileLang 编译/搜索 | 0 次新增 TileLang 编译/搜索 |
| 本次完整 block 与投影的隔离冷缓存编译后端调用 | 139 | 等待空闲窗口 |
| 本次同一磁盘缓存、新进程编译后端调用 | 0 | 等待空闲窗口 |
| 同规格权重替换及 Graph 原存储 copy_ | 结果更新、变体数不变 | 结果更新、变体数不变 |

另有 T2048 完整 block 的训练前向、prefill、逐 token decode、跨 batch 和状态
逐位对照在两种架构通过。既有相关 CPU 回归 276 项通过；本次受影响模块 CPU 回归 224 项通过。投影检查包括 FP32 原始累加与 BF16 输出逐位一致、
两种连续权重布局、三/四路、双 bias、冻结参数和未使用输出。动态 K 对照覆盖
1、17、31、32、33、63、64、65、89、90、100、127、128、129、1000、1024、4096，
另覆盖宽输出大 tile 的短归约与尾部。WKV 反向保持既有 FP32 原子累加及 VJP 容差。

“编译后端调用”在 `JITKernel._compile_and_create_adapter` 入口计数；新进程禁止
该入口，并核对输出 hash。准备后另禁止 `JITImpl.compile`、TileLang compile /
par_compile、投影代码生成入口、tuner 和 profile 读取。冷缓存使用 artifacts 下的
隔离 `TILELANG_CACHE_DIR`，正常验收保留默认缓存。

同一 1.5B 示例在两种架构上，推理报告 12 个形状族、39 个必需/可用变体；
训练报告 30 个形状族、103 个必需/可用变体。该计数描述编译完备性，
与新进程磁盘装载、进程缓存命中和实际编译后端调用分别统计。

#### 配对性能矩阵

既有完整矩阵为每种架构 28 个主工作量与一个 torch.compile 训练工作量。本次
重测受动态归约影响的训练：SM89 完成九项，SM120 完成五项。推理分别记录层、
Head 和组合延迟；表中推理采用组合延迟，包含状态重置及每序列最后一个 token 的
65536 词表 Head。训练包含全 token Head、前向、反向和梯度清零，不含优化器。
所有工作量先核对输出/state hash，再做 ABBA 配对；每次计时保留 7 组样本，
推理每组 20 次 Graph replay，训练每组 5 次。全程监控整机 GPU 进程、温度、
SM 时钟和功耗，两台机器均未检测到外部 GPU 干扰。

基线为 `a80bbbe` 执行策略及其兼容默认；SM120 另在代表完整 block 上运行一次
既有有界搜索，获得 12 个参考分派。下表“对照”取同轮基线与重放参考的较快者；
训练对照额外包含修改前的 `7dbcf52`，取三者中较快者；原始 JSON 保留全部
模式及参考覆盖记录。表中推理为代码生成保持一致的既有验收，训练为本次修正的
实际结果；待测项明确留空。时间单位为 µs。

| 模型 | 工作量 | SM89 对照 | SM89 新策略 | 变化 | SM120 对照 | SM120 新策略 | 变化 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.5B | decode B1 | 429.21 | 430.92 | +0.40% | 435.11 | 289.21 | -33.53% |
| 1.5B | decode B128 | 797.84 | 798.92 | +0.14% | 539.39 | 549.55 | +1.88% |
| 1.5B | decode B512 | 2356.67 | 2355.61 | -0.05% | 2423.48 | 1944.12 | -19.78% |
| 1.5B | prefill T128 | 556.00 | 548.81 | -1.29% | 537.71 | 429.78 | -20.07% |
| 1.5B | prefill T1024 | 1479.46 | 1476.20 | -0.22% | 1994.43 | 1429.03 | -28.35% |
| 1.5B | 训练 T128 | 5159.73 | 5122.46 | -0.72% | 4128.28 | 4151.21 | +0.56% |
| 1.5B | 训练 T1024 | 20164.81 | 19700.74 | -2.30% | 19979.82 | 19914.73 | -0.33% |
| 2.9B | decode B1 | 571.77 | 582.45 | +1.87% | 583.94 | 389.54 | -33.29% |
| 2.9B | decode B128 | 1044.40 | 1040.35 | -0.39% | 977.77 | 756.13 | -22.67% |
| 2.9B | decode B512 | 3061.48 | 3080.99 | +0.64% | 3261.16 | 2546.69 | -21.91% |
| 2.9B | prefill T128 | 702.53 | 708.83 | +0.90% | 775.61 | 538.93 | -30.52% |
| 2.9B | prefill T1024 | 1977.34 | 2024.65 | +2.39% | 2599.17 | 1773.19 | -31.78% |
| 2.9B | 训练 T128 | 6803.94 | 6756.94 | -0.69% | 5500.04 | 5480.21 | -0.36% |
| 2.9B | 训练 T1024 | 28328.42 | 27566.16 | -2.69% | 24705.99 | 24371.47 | -1.35% |
| 7.2B | decode B1 | 1055.00 | 1074.74 | +1.87% | 914.36 | 698.07 | -23.65% |
| 7.2B | decode B128 | 1797.99 | 1808.92 | +0.61% | 1622.39 | 1314.85 | -18.96% |
| 7.2B | decode B512 | 5270.63 | 5290.84 | +0.38% | 5551.69 | 4451.22 | -19.82% |
| 7.2B | prefill T128 | 1221.40 | 1232.23 | +0.89% | 1281.93 | 939.92 | -26.68% |
| 7.2B | prefill T1024 | 3929.39 | 3822.95 | -2.71% | 4720.54 | 3497.59 | -25.91% |
| 7.2B | 训练 T128 | 10733.04 | 10782.72 | +0.46% | 9068.24 | 9000.88 | -0.74% |
| 7.2B | 训练 T1024 | 42365.93 | 42201.09 | -0.39% | — | — | 待空闲窗口 |
| 13.3B | decode B1 | 1055.92 | 1072.28 | +1.55% | 913.84 | 697.17 | -23.71% |
| 13.3B | decode B128 | 1797.73 | 1805.84 | +0.45% | 1621.26 | 1321.02 | -18.52% |
| 13.3B | decode B512 | 5264.19 | 5290.55 | +0.50% | 5552.53 | 4456.51 | -19.74% |
| 13.3B | prefill T128 | 1218.66 | 1228.80 | +0.83% | 1287.13 | 939.38 | -27.02% |
| 13.3B | prefill T1024 | 3816.98 | 3798.96 | -0.47% | 4728.71 | 3494.86 | -26.09% |
| 13.3B | 训练 T128 | 10742.84 | 10796.83 | +0.50% | — | — | 待空闲窗口 |
| 13.3B | 训练 T1024 | 42071.50 | 42211.63 | +0.33% | — | — | 待空闲窗口 |

编译训练的必要独立存储复制计入计时。1.5B T128 的完整编译训练路径：

| 设备 | 对照 | 新策略 | 相对新策略 eager + Graph 的净增加 |
| --- | ---: | ---: | ---: |
| SM89 | 5276.26 | 5282.00 | 159.54 |
| SM120 | — | 等待空闲窗口 | — |

显式准备、Torch 编译特化与 Graph 捕获分别记录。推理主矩阵测量版本为
`2dc76ff`；本次训练和预热后的控制变量实验测量版本为 `a239bc3`。静态单投影
与分组投影在两种架构的代表配置上生成相同 CUDA 源码，推理策略与编译参数保持
一致；动态路径在两种架构上重新通过完整 block 的准备后零编译检查。

SM120 节点 106 在第六项训练准备后启动外部四卡训练，监控中止本轮剩余计时。
此前完成的五项配对未检测到外部 GPU 干扰；中断的 7.2B T1024 不计入结果。
尚缺 7.2B T1024、13.3B T128/T1024、1.5B 编译训练 T128，以及新进程磁盘复用
复测。项目指定的另一台 PRO 6000 八卡节点也在运行既有任务，未使用受干扰结果。

#### 目标偏差与定位

已完成的本次训练配对均满足 2% 延迟目标，并在可调收益超过测量噪声时保留至少 95%。
常规整层目标为距离最佳已测对照不超过 2%；当可调收益超过 2% 旧默认时间及
`3 × (IQR_默认 + IQR_对照)` 时，要求保留至少 95%。以下为未满足其中任一条件的整层项目；
层延迟单列，避免 Head 掩盖退化。

| 设备 | 模型 / 工作量 | scope | 延迟变化 | 可调收益保留率 |
| --- | --- | --- | ---: | ---: |
| SM89 | 1.5B / decode B1 | layer | +2.41% | 91.82% |
| SM89 | 2.9B / decode B1 | layer | +4.46% | 84.71% |
| SM89 | 2.9B / decode B512 | layer | +0.80% | 94.62% |
| SM89 | 2.9B / prefill T1024 | layer | +2.96% | 94.59% |
| SM89 | 2.9B / prefill T1024 | full | +2.39% | 96.06% |
| SM89 | 7.2B / decode B1 | layer | +3.72% | 78.51% |
| SM89 | 13.3B / decode B1 | layer | +3.36% | 83.35% |
| SM120 | 1.5B / decode B128 | full | +1.88% | 93.77% |

SM120 的 1.5B B128 Head 单列为 247.02 → 255.89 µs（+3.59%）；组合
延迟仍在 2% 内，但组合可调收益保留率为 93.77%，尚未达到 95%。

#### 动态归约的控制变量实验

两路 CMix 权重 VJP 分别计算 `M×N×K = 4096×16384×1024` 和
`16384×4096×1024`，输入为 BF16、输出包含 BF16 及 FP32 原始累加。
先固定 `(128,64,2,32)`，仅切换静态/动态 K；再独立比较流水级数和输出 tile。
每个有效候选先与静态参考逐位比较，计时前持续预热并采用 ABBA 配对。
下表为相对同轮静态 `(128,64,2,32)` 的延迟变化：

| 配置 / 改动 | SM89 value | SM89 key | SM120 value | SM120 key |
| --- | ---: | ---: | ---: | ---: |
| 原动态 K，其他参数保持相同 | +71.89% | +62.50% | +94.71% | +119.44% |
| 显式传入 K 对齐条件，其他参数保持相同 | +0.45% | +0.33% | +0.05% | +0.06% |
| 对齐已修复，改为单级流水 | +1.93% | +1.52% | +3.70% | +2.91% |
| 对齐已修复，双级流水，tile 改为 256×64 | +12.34% | +13.31% | +26.45% | +25.73% |
| 对齐已修复，双级流水，tile 改为 128×128 | -19.84% | -18.07% | -14.83% | -14.35% |

原动态 K 丢失了编译器对输入行跨度对齐的证明：X 搬运由 16 字节 cp.async
变为逐 BF16 元素加载，并增加同步。保持所有其他编译参数不变，向编译器传入
实际保证的 `K % 32 == 0` 后，生成代码恢复向量异步搬运，这两路对齐归约的额外
成本落入测量波动范围。独立 lowering 对照中，X 的标量加载语句从 2 处变为 0，
向量异步搬运从 0 处变为 2；代码与样本保存在 artifacts。

流水与 tile 也有独立影响，但幅度小于丢失向量加载的代价。32 元素拷贝保证每级
恰好覆盖一个 canonical MMA 组；对齐路径使用双级流水。两路输出几何均有足够的
CTA，128×128 tile 在这组数据中优于 256×64，因此动态权重 VJP 使用常规菜单。
选择仍由输出几何、对齐类别和设备资源决定，未增加模型名称或模型专属阈值。
每个动态 K 形状族仍只有两个变体。

#### 热权重、轮换工作集与复杂度

2560×2560、128 行投影另测热权重及 32 组独立权重组成的 400 MiB 工作集。
每次 Graph 含 32 次投影，结果折算到单次调用；权重地址和内容替换后变体数不变。
小算子同时报告绝对差异和轮次内 IQR：

| 设备 / 工作集 | 最佳已测对照 µs | 新策略 µs | 差值 µs | 变化 | 新策略 IQR µs |
| --- | ---: | ---: | ---: | ---: | ---: |
| SM89 / 热 | 14.88 | 18.07 | +3.19 | +21.46% | 0.009 |
| SM89 / 轮换 | 18.21 | 19.45 | +1.24 | +6.80% | 0.065 |
| SM120 / 热 | 24.59 | 14.66 | -9.93 | -40.38% | 0.816 |
| SM120 / 轮换 | 32.46 | 19.74 | -12.73 | -39.20% | 0.187 |

共享选择及枚举由原两套 234 行、85 个 if 语句变为 86 行、7 个 if 语句
（另有 4 个条件表达式）。历史阈值参数化形状用例从 523 个收敛到 44 个，
以设备、布局、dtype、资源、舍入、梯度与缓存行为覆盖代替配置表抄写。
相对 main，生产代码净增由 377 行降至 319 行；测试净增由 1535 行变为 1770 行。
删除旧 profile 迁移后，生产代码减少 53 行。
新增测试集中于动态 K、完整准备、失败复现、冻结路线及真实新进程缓存验证。
保留的结构特例分别对应小行数 MMA、窄输出并行度、列并行充分时的权重复用、
分组累加器预算、归约对齐与硬件同步要求，以及 owner 的真实融合/梯度语义。

汇总、原始样本、Graph 捕获成本、参考配置、遥测、训练区间分解及版本对应证明
保存在 `artifacts/cmix-acceptance-a239bc3.json`、既有
`artifacts/deterministic-acceptance-30f5098.json` 及其关联记录中，不提交生成文件。
Nsight 硬件计数器权限限制保持现状；本轮定位使用 CUDA Graph 计时事件。

### Profile 路径与离线参考

内部 `tune_linear`、`tune_norm`、`tune_tokenshift`、`tune_readout` 与 WKV tuner
继续用于未覆盖路径和离线参考。先逐位比较，再用 CUDA Graph 计时；候选只改变
输出 tile、数据搬运、流水或独立行数分组。显式 benchmark 回调只临时覆盖被测
key，不改变其他确定性分派。批量编译按实际编译选项分别组织。

公开接口 `get_tuning_cache_dir()` 返回固定根 `~/.cache/flashrwkv2`；唯一覆盖项
为导入包前设置的 `FLASHRWKV2_CACHE_DIR`。`TORCH_HOME`、`XDG_CACHE_HOME`、
release 路径和 venv 路径不影响 verified profiles。三种缓存有独立边界：

- `tuning/v2/<owner>/<fingerprint>/<key-hash>.json` 保存经过逐位验证的执行配置。
  指纹包含该 owner 的实现 AST、递归导入的本包实现依赖、Torch/TileLang/CUDA
  版本、实际 NVCC/host compiler 版本及显式编译 flags；注释、文档字符串、公共 `__init__` 包装和无关 owner 的变化不会失效全库。
  修改内核、编译选项或实现依赖会产生新指纹。驱动版本和安装路径不参与配置指纹。
- `native/` 保存 metadata/sampling 扩展。保留原有 Python/Torch ABI、工具链、
  native 源码和架构兼容键，包括 venv 身份；配置可跨 venv 复用不代表 native 二进制
  可以跨 ABI 直接复制。`python -m flashrwkv2.compile` 的 `cached` 仅描述该扩展。
- TileLang JIT 的二进制缓存由 TileLang 自身管理，保持其编译器兼容性检查与路径。
  verified profile 命中后仍可能需要首次编译选中的 kernel。

设备类别是 `(product_name, compute_major, compute_minor, SM_count)`，不含 GPU
ordinal 或 UUID。同类别多卡共享配置，4090、4090D、PRO6000 分开。
行数与 batch 按向上取整的 2 的幂分桶，最大为 512，空输入使用独立的 0 桶，
每个工作量维度最多 11 桶。模型宽度、head size、权重 shape/stride、bias dtype、
indexed 模式及 state dtype 保持精确；WKV 额外区分单 token decode 与多 token prefill。
同规格权重替换复用配置。TokenShift 与状态池搬运使用动态 batch；二分查找在
运行时迭代到区间收敛，不再按精确 batch 数生成展开代码。

服务在 capture 前使用自己的真实权重和形状运行以下公开接口：

```python
import json
from flashrwkv2 import get_tuning_cache_dir, prepare_tuning

# warmup_calls: 可重复调用的零参数函数列表，覆盖静态形状族、prefill/decode、
# capture 配置、首层/后续层、参数布局及实际梯度需求；profile 路径仍覆盖工作量桶。
# 函数负责使用独立状态或恢复状态；prepare_tuning 会实际执行每个函数一次。
coverage = prepare_tuning(warmup_calls, mode="inspect")
print(json.dumps(coverage))

# 管理员显式准备：已有有效 key 直接复用，只调缺失项。
prepared = prepare_tuning(warmup_calls, mode="tune")

# 新 serving 进程加载配置，缺失立即报出 owner、key 和 cache root。
loaded = prepare_tuning(warmup_calls, mode="require")
assert loaded["complete"]
# 此后恢复 serving state，再进行 CUDA Graph capture。
```

每个函数必须实际执行模型前向，训练函数还须执行反向，使用有效且非空的代表性输入与 metadata。
WKV 调优调用需要该 capture shape 全部 token/sequence 均处于 active 状态；
不能用零 active 的图预热占位输入进行逐位验证。
报告逐项列出 `owner/key/fingerprint/config/path/status`，其中 key 包含设备类别；
`counts` 区分 `fixed/hit/missing/tuned`，`fixed` 表示库内固定配置，
其 `fingerprint` 和 `path` 为 null；同一工作量桶中的固定配置、不同固定选择与
profile 配置分别报告。`complete` 要求实际观察到的全部配置及确定性编译变体均可用，
未观察到任何可配置 dispatch 会报错。调用方负责提供完整工作负载集合，
不能用目录内记录总数代替覆盖证明。
`inspect` 可用默认配置收集缺口，`require` 在第一项缺失时中止；普通 forward
按 owner 首次警告未调优默认配置，不自动启动调优。

缺失项不写入进程内选择缓存，后续 eager forward 能看到新发布的文件。
已加载有效配置在普通 forward 中复用；显式 prepare 每轮对每个 key 重读一次文件，
因此能观察外部进程的重新发布或删除。准备操作在 capture 内直接拒绝，未预热的缺失 key
不能在 capture 中加载。准备前已建立的 CUDA Graph 必须重新捕获。
多个进程对同一 key 通过文件锁竞争；持锁后再次读取，只有一个进程验证/调优并
用同目录临时文件原子替换发布。无效 JSON、key 或验证标记立即报错；半写入
临时文件不会被读者视为已发布 profile。
单个进程内 prepare 是独占阶段，准备前先结束其他 serving/training 调用；嵌套或
并发 prepare 会报错。准备报告也覆盖 CUDA autograd worker thread 上的反向 dispatch。

升级后使用当前版本重新执行 `prepare_tuning`。确定性形状族在准备阶段编译有限变体；
仍需 profile 的路径使用 `mode="tune"` 验证并测量缺失配置，再发布当前 fingerprint
对应的结果。旧版调优记录可直接删除。TileLang 自带编译缓存继续复用。
接口为 `prepare_tuning(calls, *, mode="require")`，不再提供 `legacy_dir` 参数或
`revalidated` 状态。移除迁移后的回归为 274 项 CPU 测试、7 项 SM89 CUDA 测试通过；
CUDA 用例覆盖当前 profile 加载、block/CMix 数值与梯度、正常调优、
新进程 profile 复用及准备后新 token 零编译。
跨 release/venv 验证应在目标环境的新进程运行 `mode="require"` 并保存报告；
修改 Torch/XDG 缓存变量仍应命中相同 profile 路径。
基准还覆盖真实 7.2B 的 `w/a=128,g=480,v=96` 首层与后续层，以及
`B=1/4/64/320/512` decode，以保证缓存 key 对应实际推理布局。

普通投影与 WKV Prepare 共用候选生成规则，每次调优最多 24 个不同配置：
先以 32 元素的 K 拷贝跨度比较两级、三级 pipeline；每种列宽只保留完整 SM
覆盖边界两侧的相邻 M tile（16–256），第一阶段最多 12 个候选。
随后只为两个经过逐位验证且计时最好的不同输出 tile 比较 64/128 拷贝跨度；
单 token/小 batch 或窄输出额外保留一个可整除所有输入宽度的最大 256–1024
跨度，第二阶段最多 12 个候选。所有候选在生成 TIR 前按共享内存上限和
整除约束剪枝，已测配置不重复扫描。最后仅对全局前三名使用较长 CUDA Graph
复测。每次拷贝仍按连续 32 元素块执行相同的 MMA 序列，保持累加顺序。
未显式调优时，投影默认配置使用共享的有限菜单与 CTA 规则；普通前向不启动搜索。

调优同时核对 FP32 累加或统计量及实际生产输出；ln 是否保存统计量、是否写回
res 等变体都必须保持相同结果。线性层的 bias presence/dtype 属于调优 key。
编译不受工具链支持的线性候选会在结果中记录 `compile_error`；运行失败直接报错。
候选的 staging buffers 超出设备共享内存上限时，在编译前直接排除。
投影候选先串行生成 TIR，再用 TileLang 按可用 CPU 核心数调度的编译线程构建并复用进程内 kernel cache。
全部构建结束后才逐一检查输出和计时；编译失败仍在对应候选中保留错误信息。
`tune_linear` 可通过 `benchmark` 回调测量实际前向阶段，结果用 `timing_scope`
区分整个阶段与单个投影。Prepare 基准先核对并调优三个分组 kernel，再按完整
前向重选 R/K/V 分组配置，保持三份 dense weights 的实际工作集；反向基准
仍独立测量分组投影。分组调优同时核对每路 BF16 结果与激活前的 FP32 累加。
CMix 前向基准同样按完整阶段重选 key/value 投影，以保留两份权重及 ReLU²
中间结果共同产生的缓存行为。
单 token 的 PostNorm+TokenShift 融合路径直接读取独立的 mix 参数，
避免每次调用都先在 GPU 上打包六路参数；CMix 的单路复用使用同一 kernel。
调优需在 CUDA Graph capture 前完成，并使用有效、非空的 active workload。
WKV 的 functional、indexed pool 模式和 state dtype 分别缓存配置；候选同时校验两种模式的
输出和所选 dtype 的最终状态，按实际调用模式计时。
WKV 候选组合 32/64 个 value columns 与 1/4 个输入 token 的共享内存预取。
四 token 预取在 head size 64/128 时展开块内 token 循环；head size 256 保留串行循环。
预取只改变输入加载与 decay 计算的排布，逐 token 的 state 更新、K 归约、BF16
舍入和每 16 token 的 FP32 checkpoint 保持原有顺序。未调优时使用单 token
预取；单步 decode 的调优也只测单 token 配置。
FP16 state 只能以 `indexed=True` 调优，逐位校验相同 FP16 舍入规则下的 functional
参考与 indexed 结果；不以降低训练精度作为调优候选。

PostNorm、TMix 各推理阶段、CMix 和 Head 线性基准可用
`--implementation tilelang|cuda-baseline` 标记
通过 `PYTHONPATH` 选中的包。两次运行复用同一基准脚本、seed、逻辑输入及尺寸，
WKV 的 `--state-dtype float16` 单独测量可选推理模式；默认仍为 `float32`。
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

NVIDIA 的 [cuBLASLt 文档](https://docs.nvidia.com/cuda/cublas/index.html#heuristics-cache)
描述了按矩阵尺寸及 GPU 选择 kernel 的启发式与缓存；这不需要应用自行枚举 tile。
[PyTorch CUDA 文档](https://docs.pytorch.org/docs/stable/notes/cuda.html)说明可分别
关闭 BF16 的低精度归约和 split-K；固定精度并不规定不同形状使用同一算法。

在 RTX 4090、PyTorch 2.14.1+cu130 和基于 `6e6dba4`、含单波次简化的验证快照上，以 seed
`20260804` 比较宽度 2048、词表 65536 的 Head。相同 BF16 输入分别以单行和
复制为 17 行调用，cuBLASLt 使用上述 workspace 设置，并关闭 BF16 低精度归约和
split-K。单行与 17 行首行的 FP32 输出有 64949/65536 个值不同，最大绝对差为
2.15e-5；对应 BF16 输出有 91/65536 个值不同。TileLang 的 FP32 和 BF16 输出
均逐位一致；17 行首行的原生输出与 TileLang 一致。cuBLASLt 调用日志显示
两种行数选择了不同算法。这个工作量的单行 FP32 Head 在 CUDA Graph 下分别
测得原生 281.68 微秒、TileLang 283.20 微秒，均为 10 批、每批 1000 次调用的
平均值；两条路径分别计时。此结果说明一般 GEMM 可交由库选择 kernel，但在
保持现有跨 batch 归约契约时，不能直接以此原生路径替换共享投影。

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
