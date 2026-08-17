# CMix FP16 inference

本文件对应 CMix 的唯一公共推理入口。它内部完成 PostNorm、TokenShift、上投影、ReLU² 和下投影，并选择 dense 或 sparse 实现。
