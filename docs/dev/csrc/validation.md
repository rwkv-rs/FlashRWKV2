# `csrc/validation.cpp`

该文件实现多个公共算子共享的 native 输入检查，包括 Tensor 的 device、dtype、
shape 与 packed-varlen metadata 契约。它只负责验证，不负责模型阶段调度。

对应的 `csrc/validation.h` 定义未注册的 native-private metadata preparation
契约及共享错误位。静态 preparation 的可选摘要参数只供 native ticket 创建
使用，不改变任何 Python API 或公开 binding；实际内容检查位于
`validation/recurrent_metadata.cu`。

`PreparedRecurrentMetadata` 返回 offsets、slots、status 和去重 workspace，
predecessor 则由独立未注册的 `build_recurrent_token_predecessor_cuda` 按需生成，
避免不使用 TokenShift 的 WKV 消费者承担准备成本。
