# Head training projection

pretrain_head_linear_bf16 接收选定 hidden rows、[vocab,C] 基座和可选 A/B，返回 logits。支持 LoRA 与每次 forward 归一化的 NoRA；输入 token 为 BF16，A/B 可为 BF16 或 FP32。投影调用共享 native-private provider，不创建完整 delta weight。
