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

一个模块要在 Python 源码、测试、基准和 C++/CUDA 源码中使用同一套模块路径。
下面的写法使用三种占位符：`<...>` 表示实际名称，`[...]` 表示可选部分，
`{a|b}` 表示只能从列出的选项中选择一个。

```text
Python 源码：./flashrwkv2/<module_name>[/<sub_module_name>]/__init__.py
测试代码：  ./tests/<module_name>[/<sub_module_name>]/test.py
性能基准：  ./benchmarks/<module_name>[/<sub_module_name>]/bench.py
C++/CUDA：  ./csrc/sm{60|75|80|90|120}/<module_name>[/<sub_module_name>]/
           {pretrain|rl_infctx|statetune|infer}_
           [{recurrent[_kda]|chunk}_]{numerical-mode}_
           {forward|backward}[_varlen].{cpp|cu}
```

具体规则如下：

- `<module_name>` 和可选的 `<sub_module_name>` 必须在四类目录中保持一致；
  没有子模块时，省略整个 `[/<sub_module_name>]`。
- CUDA 源码的第一层目录必须是 `sm60`、`sm75`、`sm80`、`sm90` 或 `sm120`，
  不要使用未列出的架构名称。
- CUDA 文件名依次表示工作负载、算法、数值模式和执行方向；工作负载只能是
  `pretrain`、`rl_infctx`、`statetune` 或 `infer`，(如果是wkv7 kernel, 算法只能是 `recurrent`、
  `recurrent_kda` 或 `chunk`)方向只能是 `forward` 或 `backward`，
  `_varlen` 表示可变长输入，扩展名只能是 `.cpp` 或 `.cu`, 两文件同名。

例如，`tmix/wkv7` 模块应按下面的方式对应组织：

```text
flashrwkv2/tmix/wkv7/__init__.py
tests/tmix/wkv7/test.py
benchmarks/tmix/wkv7/bench.py
csrc/sm90/tmix/wkv7/infer_recurrent_fp16_forward_varlen.cu
```

### Kernel 实现的阶段顺序与复用

Kernel 通常按照训练阶段的完成顺序设计：`pretrain` 和 `infer` 最早完成，
`rl_infctx`、`statetune` 等阶段稍后完成。因此，后完成阶段如果可以直接复用
早期阶段的实现，应保留自己的阶段文件名，但将文件做成指向早期真实文件的
软链接，而不是复制代码或抽取到 `csrc/common/`。

例如，下面的链接保留了 `statetune` 的构建入口，同时只维护一份实现：

```text
csrc/sm90/tmix/wkv7/pretrain_recurrent_fp16_forward.cu
csrc/sm90/tmix/wkv7/statetune_recurrent_fp16_forward.cu
    -> pretrain_recurrent_fp16_forward.cu
```

软链接目标应指向已经较早完成且行为一致的实现；如果后续阶段出现独有行为，
就解除软链接并创建独立的真实源文件。对于 kernel 实现，不要为了抽取共同部分
新增 `csrc/common/` 共享层；阶段之间的复用关系应通过目录中的软链接表达。

新增文件时，必须询问用户。

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

## 提交与 Pull Request 指南

近期提交遵循聚焦的 Conventional Commit 前缀，例如 `feat:`、`fix:`、`test:`、
`ci:`、`refactor:` 和 `style:`，也可以带 scope，例如 `feat(wkv7): ...`。
PR 描述应说明契约或性能影响，列出实际运行的测试及硬件/架构信息；发布基准
结果前必须先提供正确性证据。移植 CUDA 内核时要记录源码来源和许可证边界；
如有对应 issue，请在 PR 中关联。
