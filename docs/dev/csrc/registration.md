# `csrc/registration.cpp`

该文件创建并初始化 FlashRWKV2 的 pybind native 模块，随后把具体公共算子的
注册工作交给 `csrc/bindings.cpp`。它不拥有任何模型语义或 CUDA 实现。
