# SM90 pretrain TMix Readout forward

训练 Readout 保持权威 `train_temp` 的既有数学边界：head-wise LN、RKV residual
与 gate，返回 output projection 之前的 BF16 Tensor，并保存 backward 所需的
`mean` 与 `rstd`。Python 公共入口为 `pretrain_tmix_readout_bf16`。
