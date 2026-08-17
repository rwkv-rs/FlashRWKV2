# PostNorm FP16

公开入口只保留最终输出所需的 `infer_post_norm_output_forward_varlen`。LN、Res 和中间双输出 launcher 仅供 TMix/CMix 内部组合。
