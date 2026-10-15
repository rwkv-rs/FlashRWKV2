# BF16 training projection provider

CMix、TMix WKV Prepare、Readout 与 Head 实际共用的 native-private provider。输入为连续 CUDA BF16 token/weight，适配器为 BF16 或 FP32。返回投影结果、rank features 和归一化 A。NoRA 使用 FP32 列范数与 epsilon=1e-6，随后转为 BF16 供 GEMM 使用；低秩分支通过 GEMM 的 alpha/beta 累加。不注册 pybind。NoRA 数学来源为 Joluck/NoRA revision 4669c35ce0ffd027456272ee5050c0f58ac8d22a，CUDA 为独立 MIT 实现。
