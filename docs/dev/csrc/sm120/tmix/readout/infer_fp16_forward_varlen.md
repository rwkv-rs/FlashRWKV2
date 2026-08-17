# SM120 TMix Readout

公共入口 `infer_tmix_readout_forward_varlen` 固定执行：按 head LN、RKV residual、
gate、output projection，以及可选 output LoRA。

现有 `lnx_rkvres_xg` CUDA body 作为 Readout 内部第一阶段保留；output projection
由 owner-local dispatch 调用共享 GEMM primitive。当前实现是一个公共调用内的
多个 kernel launch，不宣称存在新的跨 LN/RKV/Linear fused CUDA kernel。
