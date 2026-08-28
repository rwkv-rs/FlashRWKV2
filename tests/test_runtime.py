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


def test_deltalog_policy_matches_pinned_albatross_exact_table() -> None:
    module = importlib.import_module("flashrwkv2.tmix.wkv7")
    expected = {
        (768, 16): 2,
        (768, 32): 3,
        (768, 64): 3,
        (768, 128): 3,
        (768, 256): 3,
        (768, 512): 3,
        (1024, 16): 2,
        (1024, 32): 3,
        (1024, 64): 3,
        (1024, 256): 3,
        (1024, 512): 3,
        (2048, 8): 2,
        (2048, 16): 3,
        (2048, 32): 3,
        (2048, 64): 3,
        (2048, 256): 3,
        (2048, 512): 4,
        (2560, 8): 2,
        (2560, 16): 3,
        (2560, 32): 3,
        (2560, 64): 3,
        (2560, 256): 3,
        (2560, 512): 4,
        (4096, 8): 2,
        (4096, 16): 3,
        (4096, 32): 3,
        (4096, 64): 3,
        (4096, 128): 3,
        (4096, 256): 3,
        (4096, 512): 4,
    }
    assert module._DELTALOG_POLICY_SOURCE_REVISION == (
        "3465da5070beceb4bab9e07b03abee1642a0bdf8"
    )
    assert module._DELTALOG_TUNED_M == expected
    fp16_unprofitable = {
        (768, 64),
        (768, 128),
        (1024, 64),
        (2048, 32),
        (2048, 64),
        (2560, 32),
        (4096, 32),
    }
    assert module._DELTALOG_FP16_UNPROFITABLE_SHAPES == fp16_unprofitable
    for (channels, batch_size), merge_interval in expected.items():
        assert (
            module._select_deltalog_merge_interval(
                channels, batch_size, 64, (12, 0), "fp32io16"
            )
            == merge_interval
        )
        expected_fp16 = (
            0 if (channels, batch_size) in fp16_unprofitable else merge_interval
        )
        assert (
            module._select_deltalog_merge_interval(
                channels, batch_size, 64, (12, 0), "fp16"
            )
            == expected_fp16
        )
    assert (
        module._select_deltalog_merge_interval(
            4096, 320, 64, (12, 0), "fp32io16"
        )
        == 0
    )
    assert (
        module._select_deltalog_merge_interval(
            4096, 64, 128, (12, 0), "fp32io16"
        )
        == 0
    )
    assert (
        module._select_deltalog_merge_interval(
            4096, 64, 64, (9, 0), "fp32io16"
        )
        == 0
    )


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
        "infer_tmix_wkv7_recurrent_deltalog_fp16_forward_varlen",
        "infer_tmix_wkv7_recurrent_deltalog_fp32io16_forward_varlen",
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
