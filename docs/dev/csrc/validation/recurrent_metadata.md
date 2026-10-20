# Packed recurrent metadata

`csrc/validation/recurrent_metadata.cpp` 注册静态/graph metadata preparation 和
private ticket。ticket 记录 metadata 的 identity、version、shape、device、stream
及 snapshot，供 TMix TokenShift、WKV 和 CMix 共享。

`csrc/validation/recurrent_metadata.cu` 在一次 CUDA 校验中检查序列端点、区间、
slot 范围与唯一性。去重使用 Fibonacci hash 和 linear probing，工作区大小由
sequence capacity 决定。状态为三个 INT32：`[error, active_tokens, active_sequences]`。

静态 preparation 回读 4 个 INT32 的摘要，得到同步异常与精确 max_seqlen。
Graph preparation 不做 D2H；每次 replay 校验 active prefix。所有 TileLang
消费者在读取有效 metadata 和写入 state 前检查 status，支持零 active warmup。

C++ ticket 的来源与 Apache-2.0 边界保留在文件头；CUDA validator 为项目实现。
