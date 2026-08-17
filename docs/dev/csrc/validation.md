# `csrc/validation.cpp`

该文件实现多个公共算子共享的 native 输入检查，包括 Tensor 的 device、dtype、
shape 与 packed-varlen metadata 契约。它只负责验证，不负责模型阶段调度。
