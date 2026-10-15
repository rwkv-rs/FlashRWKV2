# TMix training WKV Prepare backward

R/K/V 的上游梯度由现有 VRes、KK/A 等 autograd 算子汇合后传入 native，按需计算三个投影输入、基座及六个适配器张量的梯度。首层 v_first 与 value 共享图，后续层支持外部 v_first 梯度。
