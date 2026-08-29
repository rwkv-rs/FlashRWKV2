# `csrc/registration.cpp`

该文件只创建并初始化 FlashRWKV2 的私有 import-only native 模块。公共算子由
各语义所有者通过 LibTorch Stable ABI dispatcher 注册；该文件不提供 callable
Python API，也不拥有任何模型语义或 CUDA 实现。
