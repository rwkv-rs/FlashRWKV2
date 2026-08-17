# `csrc/validation/recurrent_metadata.cu`

该文件在 CUDA 侧检查 WKV recurrent packed metadata 的一致性，确保序列边界、
slot 和 elapsed-state 索引能够安全地供 recurrent kernel 使用。

它属于共享验证基础设施，不是独立 Python 算子模块。
