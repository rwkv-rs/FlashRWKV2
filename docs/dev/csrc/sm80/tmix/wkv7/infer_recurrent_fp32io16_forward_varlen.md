# `infer_recurrent_fp32io16_forward_varlen.{cpp,cu}`

该组源码实现普通 FP32IO16 recurrent native provider。它通过 `_C` 注册，供
`infer_tmix_wkv7_recurrent_fp32io16_forward_varlen` 的内部自动调度调用，不是独立
的 Python 公共接口。外部调用方只持有
`prepare_tmix_wkv7_recurrent_fp32io16_state` 返回的统一 state handle。

该 C++ 文件还注册既有 recurrent metadata ticket preparation。静态 ticket 的
内容检查只由共享 CUDA validator 执行一次，C++ 回读四个 int32 的摘要来保持
原有错误信息和最大序列长度推断。完整 metadata 的 CPU 拷贝、CPU 去重扫描和
随后重复验证已移除；静态 snapshot、ticket identity/version/stream 契约不变。
live/capacity preparation 继续全程在设备上处理内容校验，且同一 ticket 在
各层 TMix/WKV7/CMix 间共享。

Ticket 不在创建时准备 TokenShift predecessor：只在 TokenShift/CMix 首次取用
时由共享 provider 构建，静态调用及同一 capture 内的消费者复用。WKV-only
调用省去 predecessor 缓冲和 launch。默认最大长度直接由 ticket 摘要推断，
TokenShift、CMix 和 chunk 的 Python 入口不再先做差分/max/item 的重复同步。


公共入口增加可选 `output_dtype`：默认 `None` 与 token dtype 一致；FP32 输出
直接保存原有 FP32 reduction 结果。large、small-warp、KV tile 与 group4 t-loop
分别模板化输出指针/存储类型，state、输入、decay bias 预处理、归约顺序与
launch selector 不变。native binding 按传入 output tensor dtype 选择模板。

FP32 输出由普通 provider 实现；统一入口会先 materialize 所选 slot 的 pending
DeltaLog，再调用普通 provider。共享 layout validator 仅在本 binding 显式允许
FP32 输出；其他使用它的 provider 仍要求 output 与 token dtype 一致。

大值数值回归位于 `tests/tmix/wkv7/test.py`：有限 FP16 输入构造解析 WKV 输出
超过 65504 的序列，对照 FP32 解析解，检查默认路径的 FP16 溢出、两种输出的
state 一致性，以及 FP32 WKV 到 Readout 后的有限 FP16 结果。另有 FP16/BF16
普通输入、pending DeltaLog 和 live metadata CUDA Graph 的覆盖。
性能入口 `benchmarks/tmix/wkv7/bench.py --output-dtype {input,float32}` 可分别
测量默认与 FP32 输出；默认仍为 `input`。
