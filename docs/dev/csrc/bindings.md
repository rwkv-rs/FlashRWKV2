# `csrc/bindings.cpp`

该文件是 native 扩展的总注册入口。它只调用各公共语义模块提供的
`register_*_bindings`，不实现 kernel，也不把 `internal/**` helper 暴露给 Python。

`tmix/wkv7` 的普通 FP16/FP32IO16 与各自 DeltaLog launcher 虽注册在私有 `_C`
namespace，但只作为对应统一 recurrent infer 的 provider，不属于根包或模块公共
API。DeltaLog provider 同时注册统一 state handle 使用的按 slot materialize
launcher；该 symbol 同样是 `_C` 私有实现，server 只调用 handle 方法。

本次重构后，TMix 推理在这里注册 `wkv_prepare` 与 `readout` 两个最高融合岛；
CMix 仍只注册完整 `cmix` 入口。
