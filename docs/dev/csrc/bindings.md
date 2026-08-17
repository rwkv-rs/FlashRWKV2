# `csrc/bindings.cpp`

该文件是 native 扩展的总注册入口。它只调用各公共语义模块提供的
`register_*_bindings`，不实现 kernel，也不把 `internal/**` helper 暴露给 Python。

本次重构后，TMix 推理在这里注册 `wkv_prepare` 与 `readout` 两个最高融合岛；
CMix 仍只注册完整 `cmix` 入口。
