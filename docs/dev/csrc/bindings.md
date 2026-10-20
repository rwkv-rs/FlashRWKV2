# Native registrations

`csrc/bindings.cpp` 只注册 packed recurrent metadata 和 sampling。
模型算子的 TileLang 实现在对应 `flashrwkv2/<owner>/tilelang.py`，由 Python
公共入口调用；共享投影、状态搬运和归约位于 `flashrwkv2/_internal`。
