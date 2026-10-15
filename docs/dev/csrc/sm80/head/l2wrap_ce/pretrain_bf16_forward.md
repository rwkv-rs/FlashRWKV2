# Head CE/L2Wrap training

`pretrain_head_l2wrap_ce_bf16` 按 token 行分块计算 65536-way logits、CE 和
L2Wrap 梯度，避免保存所有 token 的完整 logits。可选 A/B、scaling 和 NoRA
开关使用共享 native-private BF16 projection provider，A/B 支持 BF16/FP32。

各块的 dB、dAhat 与可选 dW 在 FP32 中累加，最后执行一次 NoRA 列归一化反向，
再转换为原参数 dtype。冻结基座时不分配完整 dW，保留输入梯度。forward 保存
调用方需要的梯度，autograd backward 再应用上游 loss scaling。
