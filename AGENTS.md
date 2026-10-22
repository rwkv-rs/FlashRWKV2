# Repository Guidelines

## 核心目标

本项目为 RWKV 社区权威算子实现仓库, 为 Flash-Linear-Attention(fla) 提供高性能后端.
针对 RWKV7 模型的计算流程, 需要保证一定可维护性的同时, 找到尽可能释放硬件性能的方法.
由于 RWKV7 模型的特点, 如O(1)复杂度, 无 KV Cache等, 常规的优化方法并不适用, 
我们一般参考其它 RWKV 权威实现, 或 Kimi-Delta-Attention 实现进行优化.

## 权威 RWKV7 实现

(1) https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/rwkv_v7_numpy.py
(2) https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/run_rwkv7_qwen35.py
(3) https://github.com/BlinkDL/Albatross -- 权威底层推理引擎实现仓库 (cuda, for pro6000, 无调度, 无varlen)
(4) https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/train_temp -- 权威预训练实现仓库 (cuda, for h100)
(5) https://zhiyuan1i.github.io/posts/dplr-mathematics -- Diagonal Plus Low Rank(DPLR）的数学原理：显式转移矩阵的并行计算

## RWKV7 权重
权重一般命名规范: {arch_version}-{data_version}-{param_size}-{release_date}-{ctx_len}.pth
如: rwkv7-g1h-7.2b-20260710-ctx10240.pth
arch_version: 架构版本, 如 rwkv7(default), rwkv7a(experimental, rwkv7 with DeepEmbed), rwkv7b(experimental, rwkv7 with DeepEmbedAttn)
data_version: 数据版本, 如 g1a, g1b... (The further back in the alphabet, the better)
param_size: 参数规模, 仅有 0.1b, 0.4b, 1.5b(often used in RL), 2.9b, 7.2b(often used in the infer test), 13.3b
(1) https://huggingface.co/BlinkDL/rwkv7-g1/tree/main -- 权威权重 Release 源 (update every month)
(2) https://huggingface.co/BlinkDL/temp-latest-training-models/tree/main -- 权威权重 Test 源 (不定期update)

## 目录规范

一个模块在 Python 源码、测试和基准中使用同一套模块路径。TileLang kernel 保存在
对应 Python 模块；仍使用 C++/CUDA 的模块在 native 源码中保持同一模块路径。
下面的写法使用三种占位符：`<...>` 表示实际名称，`[...]` 表示可选部分，
`{a|b}` 表示只能从列出的选项中选择一个。

```text
Python 源码：./flashrwkv2/<module_name>[/<sub_module_name>]/__init__.py
测试代码：  ./tests/<module_name>[/<sub_module_name>]/test.py
性能基准：  ./benchmarks/<module_name>[/<sub_module_name>]/bench.py
C++/CUDA：  ./csrc/sm80/<module_name>[/<sub_module_name>]/
           {pretrain|rl_infctx|statetune|infer}_
           [{recurrent[_kda]|chunk}_]{numerical-mode}_
           {forward|backward}[_varlen].{cpp|cu}
```

具体规则如下：

- `<module_name>` 和可选的 `<sub_module_name>` 必须在四类目录中保持一致；
  没有子模块时，省略整个 `[/<sub_module_name>]`。
- CUDA 源码仅维护在 `csrc/sm80`，面向 Compute Capability 8.0 及以上设备。
  安装后的首次 CUDA 调用针对当前可见 GPU 原生编译；架构与产品差异通过同一源码中
  经过验证的执行配置表达，不新增其他架构或产品型号命名的 CUDA 源码目录。
- CUDA 文件名依次表示使用场景、算法、数值模式和执行方向；使用场景只能是
  `pretrain`、`rl_infctx`、`statetune` 或 `infer`，(如果是wkv7 kernel, 算法只能是 `recurrent`、
  `recurrent_kda` 或 `chunk`)方向只能是 `forward` 或 `backward`，
  `_varlen` 表示可变长输入，扩展名只能是 `.cpp` 或 `.cu`, 两文件同名。

例如，`tmix/wkv7` 模块应按下面的方式对应组织：

```text
flashrwkv2/tmix/wkv7/__init__.py
tests/tmix/wkv7/test.py
benchmarks/tmix/wkv7/bench.py
flashrwkv2/tmix/wkv7/tilelang.py
```

算子模块的规范名称还包括：顶层 `post_norm` 公开最终 Res 后 LN，并向融合岛
提供内部 LN/Res launcher；
`tmix/tokenshift` 负责公开的 TMix PostNorm+TokenShift 融合岛；`cmix`
负责完整 ChannelMix。CMix 的 TokenShift、ReLU²、dense/sparse projection
都是内部实现，不建立独立 Python、测试或基准模块。
SM120 TMix 推理的公共阶段固定为
`tokenshift -> wkv_prepare -> wkv7 -> readout`：`wkv_prepare` 完成 R/K/V、
WAG/WAGV、VRes 与 KK/A 准备，`readout` 完成 head LN、RKV residual、gate
和 output projection。不得以 `linear`、`activation` 等实现手段建立公共模块；
`lnx_rkvres_xg` 只允许作为 Readout 内部 kernel 的来源名称。
源码、native symbol、Python API、参数、错误信息、测试和文档中的 LN 与 Res
统一写作 `ln` 和 `res`。SM120 `T=1,C=4096` 路径使用 fused PostNorm+TokenShift kernel；其他 packed
varlen 形状由同一公共接口组合共享的 PostNorm 与 TokenShift primitive。所有路径
必须保持相同的 BF16 舍入边界和 FP32 归约顺序。

根包采用扁平导出，因此公共 Python API、公开 native symbol 和注册函数必须在
名称中保留模块所有者，不能只写算法名。例如 `tmix/wkv7` 的公共名称使用
`tmix_wkv7_recurrent` 或 `tmix_wkv7_chunk`，不能省略成裸 `recurrent` 或
`chunk`。C++/CUDA 文件已经由目录表达模块所有权，文件名仍遵循上面的阶段、
算法、数值模式和方向规范，不在文件名中重复 `tmix_wkv7`。

`csrc/sm*/internal/**` 是未注册的 native-private provider：不建立 Python、
测试或性能基准镜像，不得包含 pybind registration，也不得出现在根导出中。
这里只允许放置被两个及以上语义所有者实际复用的底层 primitive；调用者特定的
shape dispatch、调优策略和融合语义必须保留在 TMix、CMix、Head 等真实所有者中。
`internal` 不得用于兼容旧 API，也不得作为任意辅助代码的公共堆放目录。
`csrc/sm*/<owner>/internal/**` 是 owner-local native-private 实现目录，同样
不得注册 pybind，也不建立 Python、测试或 benchmark 模块；它只能被所属 owner
调用。所有真实 `.cpp/.cu`（包括两类 internal）仍须在 `docs/dev/csrc` 中镜像。
空目录不构成模块、构建入口或镜像契约。

### TileLang 数值与复用契约

模型算子使用 `flashrwkv2/<module_name>[/<sub_module_name>]/tilelang.py`，公共
Python、测试和 benchmark 的模块所有权保持一致。`flashrwkv2/_internal` 仅放
多个真实模块复用的投影、归约、状态和调优 primitive。原生源码保留 metadata
validation 与 sampling；实际 `.cpp/.cu` 仍须在 `docs/dev/csrc` 镜像。

默认训练、prefill、decode 使用同一 BF16 forward，FP32 累加与 WKV state。
可以保留显式选择的 FP16 state 推理版本，其算术仍为 FP32，不承诺与 FP32 训练逐位一致。
不得通过降低训练 state 或计算精度来实现训推逐位一致。Head
logits、所选 logprobs 与 loss 为 FP32。调优必须先逐位验证输出及 state，才计时；
不得改变每个输出的 K 归约顺序、舍入边界或使用 split-K。仅 CPU lowering 不能
作为正确性或性能验收证据，最终验收覆盖 SM89 RTX4090 与 SM120 PRO6000。

新增文件时，若当前任务没有明确授权，必须询问用户。

`setup.py` 与 `pyproject.toml` 定义构建和打包契约。不要提交 `build/`、
`artifacts/`、缓存目录或 `.egg-info` 等生成文件。

## Env
使用 uv 管理本机和远端专属环境 ./.venv, 严禁使用其它环境, 避免环境污染问题。

## Machine for Testing and Benchmarking
```bash
ssh rwkv-sha-pro6000x8
cd ~/Projects/MachineLearning/flashrwkv2
```
use git to sync your changes instead of rsync.
对于 CI/CD 已有的 build, test, bench, release 任务, 请勿手动拉起, 避免重复执行.

## 提交与 Pull Request 指南

近期提交遵循聚焦的 Conventional Commit 前缀，例如 `feat:`、`fix:`、`test:`、
`ci:`、`refactor:` 和 `style:`，也可以带 scope，例如 `feat(wkv7): ...`。
PR 描述应说明契约或性能影响，列出实际运行的测试及硬件/架构信息；发布基准
结果前必须先提供正确性证据。移植 CUDA 内核时要记录源码来源和许可证边界；
如有对应 issue，请在 PR 中关联。
