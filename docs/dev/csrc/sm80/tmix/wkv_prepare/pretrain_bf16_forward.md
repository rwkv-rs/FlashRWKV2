# TMix training WKV Prepare

pretrain_tmix_wkv_prepare_bf16 完成 R/K/V（含可选 NoRA）、WAG/WAGV、VRes 和 KK/A 准备。R/K/V native 入口保存每个投影的 rank features 与 normalized A。WAG/WAGV 由 PyTorch GEMM，VRes、A gate 和 KK/A 复用现有可微训练算子。返回 recurrence inputs、gate 和 v_first。
