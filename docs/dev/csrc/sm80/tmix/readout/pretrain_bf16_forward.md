# TMix Readout training forward

`pretrain_tmix_readout_bf16` 完成权威 `train_temp` 的 head-wise ln、RKV res
和 gate，并保存 backward 所需的 `mean` 与 `rstd`。传入 `output_weight` 时，
同一入口继续完成 output projection 及可选 NoRA；省略时返回投影前的 BF16 Tensor。

可选训练适配器接收原始 A/B、scaling 与 NoRA 开关，复用 native-private BF16
projection provider。NoRA 使用每次前向列归一化及完整归一化反向；A/B 支持 BF16/FP32。
冻结基座时跳过完整权重梯度，保留输入梯度。具体公共参数见 `docs/kernel_api.md`。
