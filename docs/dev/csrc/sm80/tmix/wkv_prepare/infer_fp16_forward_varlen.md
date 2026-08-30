# SM120 TMix WKV Prepare

对应公共 binding：`csrc/sm80/tmix/wkv_prepare/infer_fp16_forward_varlen.cpp`。

本模块只公开 `infer_tmix_wkv_prepare_forward_varlen`。它依次完成 R/K/V 投影、
W/A/G/V 低秩投影、W/G 激活、可选 VRes 和 KK/A gate，返回 WKV7 的完整输入、
Readout gate 与 `v_first`。首层通过 `v_first=None` 跳过 VRes；后续层执行 VRes。

small-row fused kernel 与 generic 多-launch 路径都属于同一个公共语义入口。
rank-in、rank-out、activation、VRes、KK/A 和基础 Linear 都是 native-private helper。

共享 GEMM primitive 位于 `csrc/sm80/internal/linear`，不属于公共 Python 模块。
