# `csrc/validation.cpp`

该文件使用 LibTorch Stable ABI 实现多个公共算子共享的 native 输入检查，包括
Tensor 的 device、dtype、shape 与 packed-varlen metadata 契约，并提供 Stable
current-stream、device guard 与 cuBLAS handle 访问。它不负责模型阶段调度。
