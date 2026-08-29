# `infer_recurrent_fp32io16_forward_varlen.{cpp,cu}`

该组源码实现普通 FP32IO16 recurrent native provider。它通过 Stable dispatcher 注册，供
`infer_tmix_wkv7_recurrent_fp32io16_forward_varlen` 的内部自动调度调用，不是独立
的 Python 公共接口。外部调用方只持有
`prepare_tmix_wkv7_recurrent_fp32io16_state` 返回的统一 state handle。
