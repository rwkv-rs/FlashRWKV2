# SPDX-License-Identifier: MIT

"""Observable package loading and public-surface behavior."""

from __future__ import annotations

import importlib
import sys

import pytest

PUBLIC_MODULES = (
    "flashrwkv2.cmix",
    "flashrwkv2.embedding",
    "flashrwkv2.head.l2wrap_ce",
    "flashrwkv2.head.linear",
    "flashrwkv2.loss.l2wrap_ce",
    "flashrwkv2.post_norm",
    "flashrwkv2.sampling",
    "flashrwkv2.tmix.a_gate",
    "flashrwkv2.tmix.kk_pre",
    "flashrwkv2.tmix.readout",
    "flashrwkv2.tmix.tokenshift",
    "flashrwkv2.tmix.vres_gate",
    "flashrwkv2.tmix.wkv_prepare",
    "flashrwkv2.tmix.wkv7",
    "flashrwkv2.tmix.wkv7.statetune",
)
PUBLIC_PREFIXES = (
    "infer_",
    "prepare_",
    "pretrain_",
    "statetune_",
    "rl_infctx_",
)


@pytest.mark.parametrize(
    ("capability", "expected"),
    (
        ((9, 0), "_C_sm90"),
        ((9, 1), "_C_sm90"),
        ((12, 0), "_C_sm120"),
        ((12, 1), "_C_sm120"),
        ((12, 9), "_C_sm120"),
    ),
)
def test_backend_selection_uses_binary_compatibility(
    capability: tuple[int, int], expected: str
) -> None:
    import flashrwkv2

    assert flashrwkv2._backend_module_name(capability) == expected


def test_backend_selection_prefers_newest_compatible_minor(monkeypatch) -> None:
    import flashrwkv2

    monkeypatch.setattr(
        flashrwkv2,
        "_NATIVE_BACKENDS",
        (*flashrwkv2._NATIVE_BACKENDS, ((12, 1), "_C_sm121")),
    )
    assert flashrwkv2._backend_module_name((12, 0)) == "_C_sm120"
    assert flashrwkv2._backend_module_name((12, 1)) == "_C_sm121"
    assert flashrwkv2._backend_module_name((12, 2)) == "_C_sm121"


@pytest.mark.parametrize("capability", ((8, 9), (10, 0), (11, 0), (13, 0)))
def test_backend_selection_rejects_incompatible_major(
    capability: tuple[int, int],
) -> None:
    import flashrwkv2

    rendered = f"sm{capability[0]}{capability[1]}"
    with pytest.raises(
        RuntimeError,
        match=rf"no binary-compatible native backend for {rendered}",
    ):
        flashrwkv2._backend_module_name(capability)


def test_backend_loader_handles_cpu_and_publishes_alias(monkeypatch) -> None:
    import flashrwkv2

    monkeypatch.setattr(flashrwkv2._torch.cuda, "is_available", lambda: False)
    assert flashrwkv2._load_native_backend() is None

    sentinel = object()
    monkeypatch.setattr(flashrwkv2._torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        flashrwkv2._torch.cuda, "get_device_capability", lambda: (12, 1)
    )
    monkeypatch.setattr(
        flashrwkv2._importlib,
        "import_module",
        lambda name: sentinel if name == "flashrwkv2._C_sm120" else None,
    )
    try:
        assert flashrwkv2._load_native_backend() is sentinel
        assert sys.modules["flashrwkv2._C"] is sentinel
    finally:
        sys.modules.pop("flashrwkv2._C", None)


def test_missing_backend_reports_capability_and_selection(monkeypatch) -> None:
    import flashrwkv2

    monkeypatch.setattr(flashrwkv2._torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        flashrwkv2._torch.cuda, "get_device_capability", lambda: (12, 1)
    )

    def missing_backend(name: str):
        raise ImportError(name)

    monkeypatch.setattr(flashrwkv2._importlib, "import_module", missing_backend)
    with pytest.raises(RuntimeError, match=r"selected backend _C_sm120 for sm121"):
        flashrwkv2._load_native_backend()


def test_root_exports_module_operators_by_identity() -> None:
    import flashrwkv2

    expected: dict[str, object] = {}
    for module_name in PUBLIC_MODULES:
        module = importlib.import_module(module_name)
        for name in module.__all__:
            if not name.startswith(PUBLIC_PREFIXES) and name != "setup_sampling_states":
                continue
            assert name not in expected, f"duplicate public operator: {name}"
            expected[name] = getattr(module, name)

    assert set(flashrwkv2.__all__) == set(expected)
    for name, operator in expected.items():
        assert getattr(flashrwkv2, name) is operator


def test_retired_public_aliases_are_absent() -> None:
    import flashrwkv2

    retired = {
        "infer_ln_forward_varlen",
        "infer_res_forward_varlen",
        "infer_post_norm_forward_varlen",
        "infer_tmix_tokenshift_forward_varlen",
        "infer_cmix_tokenshift_forward_varlen",
        "infer_cmix_relu_square_forward_varlen",
        "infer_cmix_linear_ffn_down_forward_varlen",
        "infer_tmix_vres_gate_forward_varlen",
        "infer_tmix_linear_attention_c2c_forward_varlen",
        "infer_tmix_lowrank_wag_forward_varlen",
        "infer_tmix_lowrank_wagv_vres_forward_varlen",
        "infer_tmix_kk_a_gate_forward_varlen",
        "infer_tmix_lnx_rkvres_xg_forward_varlen",
        "infer_head_linear_forward_varlen",
        "infer_recurrent_fp16_forward_varlen",
        "infer_recurrent_fp32io16_forward_varlen",
        "infer_chunk_bf16_forward_varlen",
        "prepare_recurrent_metadata",
        "pretrain_recurrent_bf16",
        "statetune_recurrent_fp32io16",
        "rl_infctx_chunk_fp32io16",
        "rl_infctx_chunk_fp32io16_factor_recompute",
    }
    assert retired.isdisjoint(flashrwkv2.__all__)
    for name in retired:
        assert not hasattr(flashrwkv2, name)
