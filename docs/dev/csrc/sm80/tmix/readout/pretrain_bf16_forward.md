# SM90 pretrain TMix Readout forward

训练 Readout 保持权威 `train_temp` 的既有数学边界：head-wise LN、RKV residual
与 gate，返回 output projection 之前的 BF16 Tensor，并保存 backward 所需的
`mean` 与 `rstd`。Python 公共入口为 `pretrain_tmix_readout_bf16`。

可选训练适配器接收原始 A/B、scaling 与 NoRA 开关，复用 native-private BF16
projection provider。NoRA 使用每次前向列归一化及完整归一化反向；A/B 支持 BF16/FP32。
冻结基座时跳过完整权重梯度，保留输入梯度。具体公共参数见 `docs/kernel_api.md`。
