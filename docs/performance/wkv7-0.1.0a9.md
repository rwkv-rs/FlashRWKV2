# FlashRWKV2 0.1.0a9 WKV7 性能提升报告

发布日期：2026-08-26

## 结论

FlashRWKV2 0.1.0a9 将 Albatross `faster3a_2607` 中自动使用的 WKV7
优化同步到现有 SM120 recurrent owner。FP32IO16 group4 t-loop 与 FP16
原生 `[K,V]` warp-pair/vector 都是对原有慢 dispatch 的内部替换，没有增加
重复的公共 kernel API。slot-native DeltaLog 是本次唯一新增的算法入口。

在正式 Quality Gate 的 RTX PRO 6000 Blackwell 测量中：

- 4 个 FP32IO16 目标 selector 点相对 0.1.0a8 的延迟全部下降，范围为
  `0.41%` 至 `19.70%`。
- 3 个 FP16 目标 selector 点相对 0.1.0a8 的延迟全部下降，范围为
  `3.03%` 至 `5.45%`。
- 5 个 candidate-only DeltaLog 完整周期均快于同一 0.1.0a9 wheel 的普通
  FP16 recurrent，周期延迟下降 `6.94%` 至 `38.64%`，对应
  `1.075x` 至 `1.630x` speedup。

这些数字只表示 WKV7 算子边界，不表示模型端到端延迟、吞吐或显存收益。

## 证据身份

| 项目 | 值 |
| --- | --- |
| Albatross 仓库 HEAD | `3465da5070beceb4bab9e07b03abee1642a0bdf8` |
| Albatross 源码提交 | `3e41bc43ed5e8332927ddd7e0ce4816cf200a6ea` |
| FlashRWKV2 candidate commit/tag | `5663621d35468e98e5aab2d500b6634aa81e72e5` / `v0.1.0a9` |
| Quality Gate | [run 32956369121](https://github.com/rwkv-rs/FlashRWKV2/actions/runs/32956369121) |
| Quality source tree | `5fddea9483dc59c150300c7ef090bdca194fa90e` |
| Baseline wheel | `flashrwkv2-0.1.0a8-cp312-cp312-manylinux_2_38_x86_64.whl` |
| Baseline wheel SHA256 | `ca1470dbe161eebb48f9bbed991d6956f91d4e354500f8733fcf3b6b4fd07a5d` |
| Candidate wheel | `flashrwkv2-0.1.0a9-cp312-cp312-manylinux_2_38_x86_64.whl` |
| Candidate wheel SHA256 | `da3fd8cbaf2e6dea8582d4c94f41524cf25af4f3e8eef18f45e35925a2426ea1` |

## 测量环境与边界

| 项目 | 值 |
| --- | --- |
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation Edition |
| Compute Capability | 12.0 / SM120 |
| Driver | 590.48.01 |
| PyTorch | 2.13.0+cu130 |
| CUDA runtime | 13.0 |
| Python | 3.12.3 |
| Token dtype | FP16，除非表格明确写 BF16 |
| FP32IO16 state dtype | FP32 |
| FP16/DeltaLog state dtype | FP16 |
| Head size | `D=64` |
| Decay bias | 启用 |

正式测量由 `benchmarks/tmix/wkv7/bench.py` 和统一的
`benchmarks/_timing.py` 完成。普通 recurrent profile 的边界是预分配 output、
state pool 和 live metadata ticket 后，直接计时连续的 native in-place launch：

- 先执行 100 次 warmup。
- 正式执行 10 个 batch，每个 batch 1,000 次，共 10,000 次计时调用。
- 使用 CUDA Event 计时，避免逐 launch 同步噪声。
- state 和 elapsed state 在每个 timing batch 前恢复，恢复操作不在 CUDA Event
  计时区间内。
- metadata 准备、tensor 分配和布局准备不在计时区间内。

DeltaLog A/B 使用同一 candidate wheel、输入、state slot 和 metadata ticket。
一个计时调用包含完整的 `M` 步周期；普通 FP16 和 DeltaLog 都执行 `M` 次
recurrent 调用，并回到各 slot 的初始 phase。因此表中的 DeltaLog 数字是完整
周期延迟，不是把单次 append 与 merge 拆开后的最佳值。

## FP32IO16：a8 与 a9 跨 wheel 比较

以下 profile 使用 `H=32,D=64`、FP16 token I/O 和 FP32 state。降低比例按
`(a8 - a9) / a8` 计算，延迟越低越好。

| 0.1.0a9 selected family / workload | 0.1.0a8 (us) | 0.1.0a9 (us) | 延迟降低 |
| --- | ---: | ---: | ---: |
| group4 VT8, `B=1,T=5` | 9.6016 | 9.5545 | 0.49% |
| group4 VT4, `B=1,T=6` | 9.7072 | 9.6563 | 0.52% |
| group4 VT8, `B=2,T=6` | 9.7794 | 9.7396 | 0.41% |
| group4 VT4, `B=1,T=10` | 12.2884 | 9.8679 | 19.70% |

`B=1,T=5/6` 和 `B=2,T=6` 的绝对收益较小，只有约 `0.04-0.05 us`，但正式
复测中方向一致且没有回退；这三点应理解为 release gate 上的无回退证据，不能
解读为跨机器都能复现的大幅提升。`B=1,T=10` 从原 large family 切到适合该
并行度的 group4 VT4 后收益最明显。

## FP16：a8 与 a9 跨 wheel 比较

以下 profile 使用 `H=64,D=64` 和物理 `[K,V]` FP16 state。新 family 是现有
公共 recurrent API 内的自动 dispatch，不是另一套公共调用方式。

| 0.1.0a9 selected family / workload | 0.1.0a8 (us) | 0.1.0a9 (us) | 延迟降低 |
| --- | ---: | ---: | ---: |
| KV warp-pair, `B=256,T=1` | 189.3399 | 183.4310 | 3.12% |
| KV vector, `B=320,T=1` | 240.5146 | 227.4087 | 5.45% |
| KV warp-pair, `B=20,T=2` | 10.2760 | 9.9645 | 3.03% |

这些结果验证了三个自动阈值：T1 的 `sequence_capacity*H=16384` 使用
warp-pair，`20480` 使用 vector，非 T1 的 `1280` 使用 warp-pair。三者都直接
访问 FlashRWKV2 既有的 `[K,V]` state，不再为 Albatross 的旧物理布局执行
shared-memory transpose。

## DeltaLog：同 wheel 完整周期 A/B

DeltaLog 没有 0.1.0a8 公共 symbol，因此不能把旧 wheel 缺少接口当成性能基线。
下表比较同一 0.1.0a9 wheel 中的普通 FP16 recurrent 与 slot-native DeltaLog。
输入 channel 数为 `C=4096`，每个活动序列一个 token。

| Workload | 普通 FP16 周期 (us) | DeltaLog 周期 (us) | 延迟降低 | Speedup |
| --- | ---: | ---: | ---: | ---: |
| `B=8,M=2` | 18.6104 | 16.1422 | 13.26% | 1.153x |
| `B=64,M=3` | 48.4213 | 45.0610 | 6.94% | 1.075x |
| `B=256,M=3` | 546.2976 | 367.5716 | 32.72% | 1.486x |
| `B=512,M=4` | 1481.8628 | 1028.4367 | 30.60% | 1.441x |
| `B=256,M=3`，slot phase 错峰 | 546.3787 | 335.2574 | 38.64% | 1.630x |

收益随 batch、merge interval 和 slot phase 分布变化，不能概括成一个固定的
“47%”数字。本次实测中 `B=64,M=3` 只有 `6.94%` 周期延迟降低，而错峰
`B=256,M=3` 达到 `38.64%`；两者都属于应保留的真实边界。

## 正确性与安全前置条件

性能数据只在下面的阻塞检查通过后被接受：

- SM120 WKV7 correctness/CUDA Graph：`96 passed, 1 deselected`。
- FP16 output/state relative RMSE 上限保持 `4.0e-3`，未放宽容差。
- compute-sanitizer memcheck：`5 passed, 92 deselected`，`0 errors`。
- compute-sanitizer racecheck：`5 passed, 92 deselected`，
  `0 hazards`、`0 errors`、`0 warnings`。
- package identity、native binary contents、native SM120 和 PTX-on-SM120
  检查全部通过。
- 发布工作流复用了上述 Quality Gate 的原始 wheel/sdist，没有重新构建发布包。

## 非目标 profile 与限制

Quality Gate 的 benchmark comparison 是 advisory，最终 summary 状态为
`warning`，而不是 `clean`。它记录了 21 个未纳入本次提升结论、candidate
比 baseline 更慢的 profile：

- 19 个 FP32IO16 BF16 profile 继续使用既有 large fallback，观测回退范围为
  `0.005%` 至 `3.406%`。最大项是 `B=1,T=64`：`43.4597 us` 到
  `44.9399 us`。
- BF16 chunk profile：`155.8807 us` 到 `156.9283 us`，回退 `0.672%`。
- BF16 pretrain forward/backward profile：`880.5633 us` 到 `882.6498 us`，
  回退 `0.237%`。

这些 profile 不属于本次自动启用的 FP16-I/O group4、FP16 `[K,V]` 或
DeltaLog family，因此报告不把它们描述为得到优化，也不声称 0.1.0a9 在所有
WKV7 shape/dtype 上无回退。发布验收结论仅覆盖上面列出的七个 selector 点和
五个 DeltaLog A/B case。

## 可复核入口

- benchmark owner：`benchmarks/tmix/wkv7/bench.py`
- 统一计时器：`benchmarks/_timing.py`
- benchmark gate：`ci/benchmark_gate.py`
- Quality Gate 工作流：`.github/workflows/pro6000-gpu.yml`
- GitHub Release：
  [FlashRWKV2 0.1.0a9](https://github.com/rwkv-rs/FlashRWKV2/releases/tag/v0.1.0a9)
- PyPI：
  [flashrwkv2 0.1.0a9](https://pypi.org/project/flashrwkv2/0.1.0a9/)
