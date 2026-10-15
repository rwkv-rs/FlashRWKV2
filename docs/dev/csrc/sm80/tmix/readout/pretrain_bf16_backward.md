# SM90 pretrain TMix Readout backward

这是 `pretrain_tmix_readout_bf16` 的 native backward，消费 forward 保存的
输入、LN 统计量与上游梯度。它不包含 output-linear backward，也不宣称与
SM120 inference Readout 具有相同的融合边界。

可选训练适配器接收原始 A/B、scaling 与 NoRA 开关，复用 native-private BF16
projection provider。NoRA 使用每次前向列归一化及完整归一化反向；A/B 支持 BF16/FP32。
冻结基座时跳过完整权重梯度，保留输入梯度。具体公共参数见 `docs/kernel_api.md`。
