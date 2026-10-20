# Native tensor validation

`csrc/validation.cpp` 提供 metadata preparation 所需的 CUDA、连续性和 device
边界检查。模型算子的形状及 dtype 检查由所属 Python 公共入口负责。
