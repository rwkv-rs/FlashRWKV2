# SM120 private Linear provider

对应源码：`csrc/sm120/internal/linear/infer_fp16_forward_varlen.cu`。

这是 TMix、CMix 与 Head 共用的 native-private FP16 GEMM provider。它包含
original/transposed layout、cuBLAS/cuBLASLt、自定义小 rows kernel 和 LowRank
内部 primitive，但不注册公共 operator，也没有 Python、测试或性能基准模块。

公共调用者负责自己的 shape dispatch 和融合语义；本目录只提供未公开的底层
launcher。generic varlen 的完整融合岛可以在一次公共调用中依次启动多个这里的
kernel。
