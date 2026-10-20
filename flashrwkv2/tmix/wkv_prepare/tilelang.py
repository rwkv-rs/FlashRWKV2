# SPDX-License-Identifier: MIT
"""One WKV preparation graph for packed inference and differentiable training."""

from ..._internal.elementwise import activation
from ..._internal.linear import adapter_linear, linear
from ..kk_pre.tilelang import prepare as prepare_key
from ..vres_gate.tilelang import vres


def prepare(
    x_r,
    x_w,
    x_k,
    x_v,
    x_a,
    x_g,
    receptance_weight,
    key_weight,
    value_weight,
    w1,
    w2,
    a1,
    a2,
    g1,
    g2,
    *,
    k_k,
    a0,
    k_a,
    w0=None,
    v_first=None,
    v0=None,
    v1=None,
    v2=None,
    head_size=64,
    adapters=((None, None, 1.0, False),) * 3,
):
    """All matrix weights use [out,in]; activations round to BF16 at each edge."""
    if (
        x_r.ndim < 2
        or x_r.numel() == 0
        or any(x.shape != x_r.shape for x in (x_w, x_k, x_v, x_a, x_g))
    ):
        raise ValueError("WKV preparation inputs must share a nonempty [...,C] shape")
    channels = x_r.shape[-1]
    if any(
        weight.shape != (channels, channels)
        for weight in (receptance_weight, key_weight, value_weight)
    ):
        raise ValueError("WKV preparation R/K/V weights must have shape [C,C]")
    lowrank_pairs = (("w", w1, w2), ("a", a1, a2), ("g", g1, g2))
    if v_first is not None:
        if v0 is None or v1 is None or v2 is None:
            raise ValueError("v0, v1, and v2 are required with v_first")
        lowrank_pairs += (("v", v1, v2),)
    for name, first, second in lowrank_pairs:
        if (
            first.ndim != 2
            or first.shape[0] == 0
            or first.shape[1] != channels
            or second.shape != (channels, first.shape[0])
        ):
            raise ValueError(f"{name} lowrank weights must have shapes [R,C] and [C,R]")
    projected = []
    for x, weight, (a, b, scale, nora) in zip(
        (x_r, x_k, x_v),
        (receptance_weight, key_weight, value_weight),
        adapters,
        strict=True,
    ):
        projected.append(adapter_linear(x, weight, a, b, scale, nora))
    r, key, value = projected
    decay = linear(activation(linear(x_w, w1), "tanh"), w2)
    if w0 is not None:
        decay = activation(decay, "identity", w0)
    if v_first is None:
        v_first = value
    else:
        value = vres(value, v_first, v0, linear(linear(x_v, v1), v2))
    rate = activation(linear(linear(x_a, a1), a2), "sigmoid", a0)
    gate = linear(activation(linear(x_g, g1), "sigmoid"), g2)
    key, recurrent_a, recurrent_b = prepare_key(key, k_k, rate, k_a, head_size)
    return r, decay, key, value, recurrent_a, recurrent_b, gate, v_first
