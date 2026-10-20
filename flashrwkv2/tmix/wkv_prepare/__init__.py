# SPDX-License-Identifier: MIT
"""Shared BF16 WKV preparation for training and packed inference."""


def _original_weight(original, runtime, name):
    if original is not None:
        return original
    if runtime is None:
        raise ValueError(f"{name} requires an original or runtime weight")
    return runtime.t()


def infer_tmix_wkv_prepare_forward_varlen(
    x_r, x_w, x_k, x_v, x_a, x_g,
    receptance_weight, key_weight, value_weight,
    w1, a1, g1, v1, w2, a2, g2, v2, v0, k_k, a0, k_a,
    *,
    v_first=None,
    w1_runtime=None, a1_runtime=None, g1_runtime=None, v1_runtime=None,
    w2_runtime=None, a2_runtime=None, g2_runtime=None, v2_runtime=None,
    receptance_lora_a=None, receptance_lora_b=None, receptance_lora_scale=1.0,
    key_lora_a=None, key_lora_b=None, key_lora_scale=1.0,
    value_lora_a=None, value_lora_b=None, value_lora_scale=1.0,
    receptance_use_nora=False, key_use_nora=False, value_use_nora=False,
    head_size=64, batch_size=1, max_seqlen=None,
):
    """Prepare packed recurrent inputs; raw decay bias is applied by WKV.

    Original weights are [out,in], runtime weights are [in,out].
    Both layouts enter the same MMA arithmetic without materializing a transpose.
    Batch and length metadata do not select different numerical implementations.
    """
    from .tilelang import prepare

    lowrank = tuple(
        _original_weight(original, runtime, name)
        for original, runtime, name in (
            (w1, w1_runtime, "w1"), (w2, w2_runtime, "w2"),
            (a1, a1_runtime, "a1"), (a2, a2_runtime, "a2"),
            (g1, g1_runtime, "g1"), (g2, g2_runtime, "g2"),
        )
    )
    return prepare(
        x_r, x_w, x_k, x_v, x_a, x_g,
        receptance_weight, key_weight, value_weight, *lowrank,
        k_k=k_k, a0=a0, k_a=k_a, v_first=v_first, v0=v0,
        v1=None if v_first is None else _original_weight(v1, v1_runtime, "v1"),
        v2=None if v_first is None else _original_weight(v2, v2_runtime, "v2"),
        head_size=head_size,
        adapters=(
            (receptance_lora_a, receptance_lora_b, receptance_lora_scale, receptance_use_nora),
            (key_lora_a, key_lora_b, key_lora_scale, key_use_nora),
            (value_lora_a, value_lora_b, value_lora_scale, value_use_nora),
        ),
    )


def pretrain_tmix_wkv_prepare_bf16(
    x_r, x_w, x_k, x_v, x_a, x_g,
    receptance_weight, key_weight, value_weight,
    w0, w1, w2, a0, a1, a2, g1, g2, k_k, k_a,
    *,
    v_first=None, v0=None, v1=None, v2=None,
    receptance_lora_a=None, receptance_lora_b=None,
    key_lora_a=None, key_lora_b=None, value_lora_a=None, value_lora_b=None,
    receptance_lora_scale=1.0, key_lora_scale=1.0, value_lora_scale=1.0,
    receptance_use_nora=False, key_use_nora=False, value_use_nora=False,
    head_size=64,
):
    """Differentiable preparation using the same BF16 edges as inference."""
    from .tilelang import prepare

    return prepare(
        x_r, x_w, x_k, x_v, x_a, x_g,
        receptance_weight, key_weight, value_weight,
        w1.t(), w2.t(), a1.t(), a2.t(), g1.t(), g2.t(),
        w0=w0, k_k=k_k, a0=a0, k_a=k_a, v_first=v_first, v0=v0,
        v1=None if v1 is None else v1.t(),
        v2=None if v2 is None else v2.t(),
        head_size=head_size,
        adapters=(
            (receptance_lora_a, receptance_lora_b, receptance_lora_scale, receptance_use_nora),
            (key_lora_a, key_lora_b, key_lora_scale, key_use_nora),
            (value_lora_a, value_lora_b, value_lora_scale, value_use_nora),
        ),
    )


__all__ = ["pretrain_tmix_wkv_prepare_bf16", "infer_tmix_wkv_prepare_forward_varlen"]
