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
