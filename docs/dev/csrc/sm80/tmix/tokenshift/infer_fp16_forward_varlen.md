# TMix PostNorm and TokenShift

公共入口统一表达 Res、LN、六路 TokenShift 和 shift-state 更新。`B=1,T=1,C=4096` 使用真正 fused kernel，其他 packed varlen 形状在入口内部顺序 launch。

默认 `max_seqlen` 由静态 ticket preparation 推断，不在 Python 侧预先做设备
reduction 或 `.item()`。TokenShift predecessor 由 ticket 首次取用时构建并复用；
没有 ticket 的 native 调用在 metadata 校验后显式调用同一个 private launcher。
