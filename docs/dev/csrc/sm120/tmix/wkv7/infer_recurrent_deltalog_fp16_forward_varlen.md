# `infer_recurrent_deltalog_fp16_forward_varlen.{cpp,cu}`

该组源码实现 DeltaLog FP16 recurrent native provider。它通过 Stable dispatcher 注册，仅供
`infer_tmix_wkv7_recurrent_fp16_forward_varlen` 在内部 policy 命中时调用，不构成
Python 公共接口。`M`、phase 和 logs 均由统一 FP16 state handle 私有管理；下游
不得把该 launcher 当作第二种用户调用方式。同一 private provider 还实现按 slot
materialize kernel，供 handle 的显式物化与统一入口回退普通 FP16 前使用；它复用
预分配 status workspace，不在热路径分配显存。
