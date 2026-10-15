# Head training projection backward

接收任意上游 dlogits，按 autograd 梯度标志返回 hidden、基座、A、B 的梯度。用于 FTPO 等调用方自定义目标；冻结基座时不分配词表大小的 dW。
