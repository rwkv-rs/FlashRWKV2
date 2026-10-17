# SM120 TMix Readout

公共入口 `infer_tmix_readout_forward_varlen` 固定执行：按 head LN、RKV residual、
gate、output projection，以及可选 output LoRA。

现有 `lnx_rkvres_xg` CUDA body 作为 Readout 内部第一阶段保留；output projection
由 owner-local dispatch 调用共享 GEMM primitive。当前实现是一个公共调用内的
多个 kernel launch，不宣称存在新的跨 LN/RKV/Linear fused CUDA kernel。


WKV 输入可为 FP16 或 FP32；其余输入/权重仍为 FP16。basic、warp 与
`D=128/256` generic body 将 WKV 加载类型模板化，FP32 输入不经过 half2
转换，直接以 float/float2 读取并完成 head LN。原有 reduction、shape selector
和 RKV residual/gate 运算保持不变。第一阶段显式分配 FP16 prelinear tensor，
projection、可选 output LoRA 和公共返回值仍为 FP16。

`tests/tmix/readout/test.py` 覆盖两种 WKV dtype、三种 head size、basic/warp
任务量以及超过 FP16 范围的输入；CPU 数值 fixture 对照 FP64 LN 参考。
`benchmarks/tmix/readout/bench.py --wkv-dtype {float16,float32}` 对应两种推理
输入精度，默认仍为 `float16`。
