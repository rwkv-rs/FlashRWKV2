# `infer_recurrent_deltalog_fp32io16_forward_varlen.{cpp,cu}`

该组源码实现 FP32 state、FP16 token IO 的 DeltaLog recurrent native provider。
它通过 Stable dispatcher 注册，仅供 `infer_tmix_wkv7_recurrent_fp32io16_forward_varlen` 在内部
policy 命中时调用，不构成 Python 公共接口。`M`、phase 和 FP32 logs 均由统一
FP32IO16 state handle 私有管理；下游不得把该 launcher 当作第二种用户调用方式。
同一 private provider 还实现按 slot materialize kernel，供 handle 的显式物化与
统一入口回退普通 FP32IO16 前使用；它复用预分配 status workspace，不在热路径
分配显存。
