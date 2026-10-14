# CMix FP16 inference

本文件对应 CMix 的唯一公共推理入口。它内部完成 PostNorm、TokenShift、上投影、ReLU² 和下投影，并选择 dense 或 sparse 实现。

默认 `max_seqlen` 由静态 ticket preparation 推断，不在 Python 侧预先做设备
reduction 或 `.item()`。TokenShift predecessor 由 ticket 首次取用时构建并复用；
没有 ticket 的 native 调用在 metadata 校验后显式调用同一个 private launcher。
