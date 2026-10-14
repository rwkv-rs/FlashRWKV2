# `csrc/validation/recurrent_metadata.cu`

该文件在 CUDA 侧检查 WKV recurrent packed metadata 的一致性，确保序列边界、
slot 和 elapsed-state 索引能够安全地供 recurrent kernel 使用。

它属于共享验证基础设施，不是独立 Python 算子模块。

每份 metadata 由一次 GPU 内容校验完成序列端点、区间、slot 范围及唯一性检查。
去重工作区采用 Fibonacci hash 与 linear probing，表容量为不小于两倍 sequence
capacity 的最小二次幂，不再分配或清空整个 state pool 大小的数组。校验结果仍为
三个 int32：`[error, active_tokens, active_sequences]`，之后生成 TokenShift 所需的
token predecessor；所有消费者在状态写入前检查同一个结果。

静态 ticket preparation 可以向该次校验传入内部的四个 int32 摘要缓冲区，返回
`[error, actual_max_seqlen, first_offset, final_offset]`。主机仅回读这 16 字节，
保持同步异常和精确 `max_seqlen` 推断，不再回读完整 offsets/slots 或重复进行
CPU 内容扫描。校验与摘要使用同一份 GPU snapshot。

live/capacity ticket 不生成主机摘要，不进行 D2H；每次 CUDA Graph replay 校验
当前 active metadata 一次，并由整套 stateful operators 共享结果。
