# TMix PostNorm and TokenShift

公共入口统一表达 Res、LN、六路 TokenShift 和 shift-state 更新。`B=1,T=1,C=4096` 使用真正 fused kernel，其他 packed varlen 形状在入口内部顺序 launch。
