# BF16 training projection backward

按调用方的梯度需求计算 dX/dW/dA/dB。冻结基座省去 dW。A/B GEMM 使用 FP32 梯度输出；NoRA 列归一化反向使用 FP32 归约，零范数列采用 PyTorch 的零范数导数约定。返回适配器原始 dtype 的梯度。Head CE 可跨块累加 dAhat 后仅执行一次归一化反向。本 provider 不注册 pybind。
