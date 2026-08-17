# SM90 pretrain TMix Readout backward

这是 `pretrain_tmix_readout_bf16` 的 native backward，消费 forward 保存的
输入、LN 统计量与上游梯度。它不包含 output-linear backward，也不宣称与
SM120 inference Readout 具有相同的融合边界。
