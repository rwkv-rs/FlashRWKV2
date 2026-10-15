
可选训练适配器接收原始 A/B、scaling 与 NoRA 开关，复用 native-private BF16
projection provider。NoRA 使用每次前向列归一化及完整归一化反向；A/B 支持 BF16/FP32。
冻结基座时跳过完整权重梯度，保留输入梯度。具体公共参数见 `docs/kernel_api.md`。
