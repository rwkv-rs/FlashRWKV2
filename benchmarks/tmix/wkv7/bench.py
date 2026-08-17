# SPDX-License-Identifier: MIT

"""Benchmark the raw RWKV-7 FP32-state recurrent inference operator.

This benchmark owns only operator shapes and packed workloads.  It does not
define a model, compose an operator graph, or multiply any model-level
quantity into the measured latency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch

import flashrwkv2


DECAY_RATE = 0.6065306597126334


@dataclass(frozen=True, slots=True)
class ImportedSourceFamily:
    name: str
    repository: str
    revision: str
    license: str
    paths: tuple[str, ...]


SOURCE_FAMILY = ImportedSourceFamily(
    name="albatross-faster3a-2607-recurrent-fp32io16",
    repository="https://github.com/BlinkDL/Albatross",
    revision="ee3308f6922e59f2166c7fac3c5a192340a2b48e",
    license="Apache-2.0",
    paths=(
        "faster3a_2607/cuda/rwkv7_wkv_fp32_v2.cu",
        "csrc/sm120/tmix/wkv7/infer_recurrent_fp32io16_forward_varlen.cu",
        "csrc/sm120/tmix/wkv7/infer_recurrent_fp32io16_forward_varlen.cpp",
    ),
)


@dataclass(frozen=True, slots=True)
class KernelSpec:
    provider: str
    name: str
    layouts: tuple[str, ...]
    state_dtype: str
    token_dtypes: tuple[str, ...]
    head_sizes: tuple[int, ...]
    source_family: str


KERNEL_SPEC = KernelSpec(
    provider="Albatross",
    name="infer_tmix_wkv7_recurrent_fp32io16_forward_varlen",
    layouts=("packed",),
    state_dtype="float32",
    token_dtypes=("float16", "bfloat16"),
    head_sizes=(64, 128, 256),
    source_family=SOURCE_FAMILY.name,
)


def rwkv7_decay_logits_reference(
    r: torch.Tensor,
    decay_logits: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    state_pool: torch.Tensor,
    cu_seqlens: torch.Tensor,
    state_indices: torch.Tensor,
    scale: float = 1.0,
    decay_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent FP32 oracle used only by the benchmark correctness gate."""

    if r.ndim != 3:
        raise ValueError("reference expects packed inputs with shape [total_tokens,H,D]")
    total_tokens, num_heads, head_size = r.shape
    offsets = tuple(int(value) for value in cu_seqlens.cpu().tolist())
    slots = tuple(int(value) for value in state_indices.cpu().tolist())
    state_pool = state_pool.float().clone()

    bias = None if decay_bias is None else decay_bias.reshape(num_heads, head_size)
    output = torch.empty(
        (total_tokens, num_heads, head_size),
        device=r.device,
        dtype=torch.float32,
    )
    r_f32 = r.float()
    logits_f32 = decay_logits.float()
    k_f32 = k.float()
    v_f32 = v.float()
    a_f32 = a.float()
    b_f32 = b.float()

    for sequence_index, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        state = state_pool[slots[sequence_index]].clone()
        for token_index in range(start, end):
            token_logits = logits_f32[token_index]
            if bias is not None:
                token_logits = token_logits + bias.float()
            retention = torch.exp(-DECAY_RATE * torch.sigmoid(token_logits))
            a_state = torch.einsum("hk,hkv->hv", a_f32[token_index], state)
            state = (
                retention.unsqueeze(-1) * state
                + b_f32[token_index].unsqueeze(-1) * a_state.unsqueeze(-2)
                + k_f32[token_index].unsqueeze(-1)
                * v_f32[token_index].unsqueeze(-2)
            )
            output[token_index] = float(scale) * torch.einsum(
                "hk,hkv->hv", r_f32[token_index], state
            )
        state_pool[slots[sequence_index]] = state

    return output, state_pool


ALBATROSS_BT_MATRIX = (
    (1, 1),
    (1, 2),
    (1, 4),
    (1, 8),
    (1, 16),
    (1, 32),
    (1, 64),
    (1, 128),
    (1, 256),
    (2, 1),
    (4, 1),
    (8, 1),
    (16, 1),
    (32, 1),
    (64, 1),
    (128, 1),
    (256, 1),
    (2, 2),
    (4, 4),
    (8, 8),
    (16, 16),
)


@dataclass(frozen=True, slots=True)
class OperatorShape:
    name: str
    num_heads: int
    head_size: int = 64

    @property
    def channels(self) -> int:
        return self.num_heads * self.head_size


OPERATOR_SHAPES = {
    "h32d64": OperatorShape("h32d64", 32),
    "h40d64": OperatorShape("h40d64", 40),
    "h64d64": OperatorShape("h64d64", 64),
}


STRESS_CASES = {
    "stress_decode_b320_t1": (1,) * 320,
    "stress_decode_b2048_t1": (1,) * 2048,
    "stress_equal_b320_t16": (16,) * 320,
    "stress_ragged_b320_t1_to_t16": tuple(range(1, 17)) * 20,
    "stress_ragged_long_b32": (1, 4, 8, 16, 32, 64, 96, 128) * 4,
    "stress_ragged_skew_b32": (128,) + (1,) * 31,
}


@dataclass(frozen=True, slots=True)
class Workload:
    label: str
    lengths: tuple[int, ...]

    @property
    def batch_size(self) -> int:
        return len(self.lengths)

    @property
    def total_tokens(self) -> int:
        return sum(self.lengths)

    @property
    def uniform_t(self) -> int | None:
        if not self.lengths or len(set(self.lengths)) != 1:
            return None
        return self.lengths[0]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _sha256_paths(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _native_source_set_hash(root: Path) -> str:
    relative_paths = (
        "csrc/bindings.cpp",
        "csrc/bindings.h",
        "csrc/registration.cpp",
        "csrc/validation.cpp",
        "csrc/validation.h",
        "csrc/validation/recurrent_metadata.cu",
        "csrc/sm120/tmix/wkv7/infer_recurrent_fp32io16_forward_varlen.cu",
        "csrc/sm120/tmix/wkv7/infer_recurrent_fp32io16_forward_varlen.cpp",
        "csrc/sm120/tmix/wkv7/recurrent_decay.cuh",
    )
    return _sha256_paths(root / relative for relative in relative_paths)


def _git_metadata(root: Path) -> dict[str, str]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            args,
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    return {
        "revision": run("git", "rev-parse", "HEAD"),
        "status": run("git", "status", "--short"),
    }


def _hardware_metadata() -> dict[str, object]:
    capability = torch.cuda.get_device_capability()
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {
        "name": torch.cuda.get_device_name(),
        "compute_capability": f"{capability[0]}.{capability[1]}",
        "total_memory_bytes": properties.total_memory,
    }


def _software_metadata() -> dict[str, str]:
    return {
        "torch": torch.__version__,
        "cuda_runtime": str(torch.version.cuda),
        "python": platform.python_version(),
    }


def _parse_case(value: str) -> Workload:
    if value in STRESS_CASES:
        return Workload(value, STRESS_CASES[value])
    try:
        batch_text, tokens_text = value.lower().split("x", 1)
        batch_size = int(batch_text)
        tokens = int(tokens_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"case must be BxT, for example 4x16; got {value!r}"
        ) from error
    if batch_size <= 0 or tokens <= 0:
        raise argparse.ArgumentTypeError("B and T must be positive")
    return Workload(value, (tokens,) * batch_size)


def default_workloads(include_stress: bool) -> tuple[Workload, ...]:
    workloads = tuple(
        Workload(f"{batch_size}x{tokens}", (tokens,) * batch_size)
        for batch_size, tokens in ALBATROSS_BT_MATRIX
    )
    if include_stress:
        workloads += tuple(
            Workload(label, lengths) for label, lengths in STRESS_CASES.items()
        )
    return workloads


def _dtype_from_name(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16}[name]


def _relative_rmse(actual: torch.Tensor, expected: torch.Tensor) -> float:
    difference = actual.float() - expected.float()
    baseline = expected.float().square().mean().sqrt().clamp_min(1.0e-6)
    return float((difference.square().mean().sqrt() / baseline).item())


def _max_abs(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float((actual.float() - expected.float()).abs().max().item())


def _make_inputs(
    workload: Workload,
    operator_shape: OperatorShape,
    dtype: torch.dtype,
    *,
    device: torch.device,
    seed: int,
    with_decay_bias: bool,
) -> dict[str, torch.Tensor | None]:
    generator = torch.Generator(device=device).manual_seed(seed)
    total_tokens = workload.total_tokens
    shape = (total_tokens, operator_shape.num_heads, operator_shape.head_size)

    def random(scale: float) -> torch.Tensor:
        return (
            torch.randn(shape, device=device, dtype=torch.float32, generator=generator)
            .mul(scale)
            .to(dtype)
        )

    offsets = [0]
    for length in workload.lengths:
        offsets.append(offsets[-1] + length)
    cu_seqlens = torch.tensor(offsets, device=device, dtype=torch.int32)
    state_indices = torch.arange(
        workload.batch_size - 1,
        -1,
        -1,
        device=device,
        dtype=torch.int32,
    )
    state_slots = workload.batch_size + 4
    state_pool = torch.randn(
        (
            state_slots,
            operator_shape.num_heads,
            operator_shape.head_size,
            operator_shape.head_size,
        ),
        device=device,
        dtype=torch.float32,
        generator=generator,
    ).mul_(0.02)
    decay_bias = None
    if with_decay_bias:
        decay_bias = torch.randn(
            (operator_shape.num_heads, operator_shape.head_size),
            device=device,
            dtype=dtype,
            generator=generator,
        ).mul_(0.25)
    return {
        "r": random(0.15),
        "decay_logits": random(4.0),
        "k": random(0.08),
        "v": random(0.08),
        "a": random(0.08),
        "b": random(0.08),
        "cu_seqlens": cu_seqlens,
        "state_indices": state_indices,
        "state_pool": state_pool,
        "decay_bias": decay_bias,
    }


def _run_public_correctness(
    inputs: dict[str, torch.Tensor | None],
    *,
    limits: dict[str, float],
) -> dict[str, object]:
    r = inputs["r"]
    decay_logits = inputs["decay_logits"]
    k = inputs["k"]
    v = inputs["v"]
    a = inputs["a"]
    b = inputs["b"]
    cu_seqlens = inputs["cu_seqlens"]
    state_indices = inputs["state_indices"]
    state_pool = inputs["state_pool"]
    decay_bias = inputs["decay_bias"]
    assert isinstance(r, torch.Tensor)
    assert isinstance(decay_logits, torch.Tensor)
    assert isinstance(k, torch.Tensor)
    assert isinstance(v, torch.Tensor)
    assert isinstance(a, torch.Tensor)
    assert isinstance(b, torch.Tensor)
    assert isinstance(cu_seqlens, torch.Tensor)
    assert isinstance(state_indices, torch.Tensor)
    assert isinstance(state_pool, torch.Tensor)

    expected_output, expected_state = rwkv7_decay_logits_reference(
        r,
        decay_logits,
        k,
        v,
        a,
        b,
        state_pool=state_pool,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        decay_bias=decay_bias,
    )
    ticket = flashrwkv2.prepare_tmix_wkv7_recurrent_metadata(
        cu_seqlens,
        state_indices,
        total_tokens=r.shape[0],
        state_pool_size=state_pool.shape[0],
    )
    observed_state = state_pool.clone()
    observed_output = flashrwkv2.infer_tmix_wkv7_recurrent_fp32io16_forward_varlen(
        r,
        decay_logits,
        k,
        v,
        a,
        b,
        state_pool=observed_state,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        decay_bias=decay_bias,
        validated_metadata=ticket,
    )
    torch.cuda.synchronize()

    active = state_indices.long()
    expected_active = expected_state.index_select(0, active)
    observed_active = observed_state.index_select(0, active)
    output_rmse = _relative_rmse(observed_output, expected_output)
    state_rmse = _relative_rmse(observed_active, expected_active)
    output_max_abs = _max_abs(observed_output, expected_output)
    state_max_abs = _max_abs(observed_active, expected_active)
    selected = set(int(slot) for slot in state_indices.cpu().tolist())
    untouched = [
        index
        for index in range(state_pool.shape[0])
        if index not in selected
    ]
    untouched_ok = (
        True
        if not untouched
        else torch.equal(
            observed_state.index_select(
                0,
                torch.tensor(untouched, device=state_pool.device, dtype=torch.long),
            ),
            state_pool.index_select(
                0,
                torch.tensor(untouched, device=state_pool.device, dtype=torch.long),
            ),
        )
    )

    # A second launch with an identical reset state is the deterministic check.
    deterministic_state = state_pool.clone()
    deterministic_output = flashrwkv2.infer_tmix_wkv7_recurrent_fp32io16_forward_varlen(
        r,
        decay_logits,
        k,
        v,
        a,
        b,
        state_pool=deterministic_state,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        decay_bias=decay_bias,
        validated_metadata=ticket,
    )
    torch.cuda.synchronize()
    deterministic = torch.equal(observed_output, deterministic_output) and torch.equal(
        observed_state, deterministic_state
    )
    finite = bool(
        torch.isfinite(observed_output).all()
        and torch.isfinite(observed_active).all()
    )
    passed = bool(
        finite
        and output_rmse <= limits["output_relative_rmse"]
        and state_rmse <= limits["state_relative_rmse"]
        and untouched_ok
        and deterministic
    )
    return {
        "passed": passed,
        "output_relative_rmse": output_rmse,
        "state_relative_rmse": state_rmse,
        "output_max_abs": output_max_abs,
        "state_max_abs": state_max_abs,
        "limits": limits,
        "finite": finite,
        "deterministic": deterministic,
        "untouched_slots": untouched_ok,
    }


def _flatten_inputs(
    inputs: dict[str, torch.Tensor | None],
) -> tuple[torch.Tensor, ...]:
    tensors = tuple(inputs[name] for name in ("r", "decay_logits", "k", "v", "a", "b"))
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise TypeError("benchmark inputs are incomplete")
    return tuple(tensor for tensor in tensors)


def _timed_native_launch(
    flat_inputs: tuple[torch.Tensor, ...],
    inputs: dict[str, torch.Tensor | None],
    *,
    output: torch.Tensor,
    state: torch.Tensor,
    ticket: object,
    scale: float = 1.0,
) -> None:
    cu_seqlens = inputs["cu_seqlens"]
    state_indices = inputs["state_indices"]
    decay_bias = inputs["decay_bias"]
    assert isinstance(cu_seqlens, torch.Tensor)
    assert isinstance(state_indices, torch.Tensor)
    extension = flashrwkv2._C
    if extension is None:
        raise RuntimeError("flashrwkv2._C is not loaded")
    extension.tmix_wkv7_recurrent_fp32_from_decay_logits(
        cu_seqlens,
        state_indices,
        state,
        *flat_inputs,
        output,
        scale,
        decay_bias=decay_bias,
        validated_metadata=ticket,
    )


def _measure(
    flat_inputs: tuple[torch.Tensor, ...],
    inputs: dict[str, torch.Tensor | None],
    *,
    warmup: int,
    samples: int,
    sample_iters: int,
    ticket: object,
) -> list[float]:
    reset_state = inputs["state_pool"]
    if not isinstance(reset_state, torch.Tensor):
        raise TypeError("benchmark state pool is incomplete")
    output = torch.empty_like(flat_inputs[3])
    state = reset_state.clone()

    for _ in range(warmup):
        state.copy_(reset_state)
        for _ in range(sample_iters):
            _timed_native_launch(
                flat_inputs, inputs, output=output, state=state, ticket=ticket
            )
    torch.cuda.synchronize()

    measurements: list[float] = []
    for _ in range(samples):
        state.copy_(reset_state)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(sample_iters):
            _timed_native_launch(
                flat_inputs, inputs, output=output, state=state, ticket=ticket
            )
        end.record()
        end.synchronize()
        measurements.append(start.elapsed_time(end) / sample_iters)
    return measurements


def _percentile(values: list[float], quantile: float) -> float:
    if not values or not 0.0 <= quantile <= 1.0:
        raise ValueError("percentile requires non-empty samples and q in [0, 1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _measure_row(
    measurements: list[float],
    workload: Workload,
) -> dict[str, object]:
    p50 = _percentile(measurements, 0.5)
    return {
        "raw_latency_ms": measurements,
        "p10_ms": _percentile(measurements, 0.1),
        "p50_ms": p50,
        "p90_ms": _percentile(measurements, 0.9),
        "tok_s_p50": workload.total_tokens * 1000.0 / p50,
    }


def _use_small_fp32(batch_size: int, max_seqlen: int, io_fp16: bool) -> bool:
    if io_fp16:
        return (
            (max_seqlen == 1 and batch_size <= 96)
            or (max_seqlen == 2 and batch_size <= 21)
            or (max_seqlen == 3 and batch_size <= 3)
            or (max_seqlen == 4 and batch_size in (1, 3))
            or (batch_size == 1 and 5 <= max_seqlen <= 11)
        )
    return (
        max_seqlen == 1
        or (max_seqlen == 2 and batch_size <= 96)
        or (max_seqlen == 3 and (batch_size <= 4 or batch_size == 6))
        or (max_seqlen == 4 and batch_size in (1, 3))
        or (batch_size == 1 and 5 <= max_seqlen <= 9)
    )


def _selected_fp32_family(workload: Workload, dtype: torch.dtype) -> str:
    """Mirror the native Albatross-family policy for benchmark attribution."""

    return (
        "wkv_fp32_v2_small_warp"
        if _use_small_fp32(workload.batch_size, max(workload.lengths), dtype == torch.float16)
        else "wkv_fp32_v2"
    )


def _load_limits(root: Path) -> dict[str, float]:
    fixture = root / "tests/fixtures/tolerances-v1.json"
    payload = json.loads(fixture.read_text(encoding="utf-8"))
    return {
        "output_relative_rmse": float(
            payload["fp32io16_recurrent"]["output_relative_rmse"]
        ),
        "state_relative_rmse": float(
            payload["fp32io16_recurrent"]["state_relative_rmse"]
        ),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shapes",
        nargs="+",
        choices=tuple(OPERATOR_SHAPES),
        default=list(OPERATOR_SHAPES),
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=None,
        help="B x T cases such as 1x1 16x16; defaults to the 21-case matrix",
    )
    parser.add_argument(
        "--dtype",
        nargs="+",
        choices=("bfloat16", "float16"),
        default=["bfloat16"],
    )
    parser.add_argument("--stress", action="store_true")
    parser.add_argument("--decay-bias", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument(
        "--sample-iters",
        type=int,
        default=4,
        help="consecutive in-place launches averaged into each timing sample",
    )
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/wkv7-recurrent-fp32io16.json"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("the recurrent benchmark requires a CUDA device")
    if args.warmup < 0 or args.samples <= 0 or args.sample_iters <= 0:
        raise ValueError("warmup must be non-negative and timing counts positive")

    root = _repo_root()
    workloads = (
        tuple(_parse_case(value) for value in args.cases)
        if args.cases is not None
        else default_workloads(args.stress)
    )
    family = SOURCE_FAMILY
    spec = KERNEL_SPEC
    limits = _load_limits(root)
    git = _git_metadata(root)
    extension = getattr(flashrwkv2, "_C", None)
    if extension is None:
        raise RuntimeError("flashrwkv2._C is not loaded; build the CUDA extension first")
    extension_path = Path(getattr(extension, "__file__", ""))
    payload: dict[str, object] = {
        "schema_version": 1,
        "benchmark": "flashrwkv2_wkv7_recurrent_fp32io16",
        "revision": git["revision"],
        "git_status": git["status"],
        "benchmark_script_sha256": _sha256_paths((Path(__file__).resolve(),)),
        "native_source_set_sha256": _native_source_set_hash(root),
        "compiled_extension": {
            "path": str(extension_path),
            "sha256": _sha256_paths((extension_path,))
            if extension_path.is_file()
            else None,
        },
        "upstream": asdict(family),
        "kernel": asdict(spec),
        "hardware": _hardware_metadata(),
        "software": _software_metadata(),
        "configuration": {
            "operator_shapes": args.shapes,
            "dtype": args.dtype,
            "state_dtype": "float32",
            "stress": args.stress,
            "decay_bias": args.decay_bias,
            "warmup": args.warmup,
            "samples": args.samples,
            "sample_iters": args.sample_iters,
            "seed": args.seed,
            "correctness_limits": limits,
        },
        "results": [],
    }

    device = torch.device("cuda")
    results = payload["results"]
    assert isinstance(results, list)
    for dtype_name in args.dtype:
        dtype = _dtype_from_name(dtype_name)
        for shape_name in args.shapes:
            operator_shape = OPERATOR_SHAPES[shape_name]
            for workload_index, workload in enumerate(workloads):
                base = {
                    "operator_shape": operator_shape.name,
                    "channels": operator_shape.channels,
                    "B": workload.batch_size,
                    "T": workload.uniform_t,
                    "lengths": workload.lengths,
                    "case": workload.label,
                    "num_heads": operator_shape.num_heads,
                    "head_size": operator_shape.head_size,
                    "token_dtype": dtype_name,
                    "state_dtype": "float32",
                    "boundary": "preallocated_native_consecutive_in_place_launches",
                    "selected_kernel_family": _selected_fp32_family(workload, dtype),
                    "dispatch": {
                        "batch_size": workload.batch_size,
                        "max_seqlen": max(workload.lengths),
                        "policy": (
                            "automatic policy uses Albatross large/small-auto; "
                            "upstream forced short-block is retained under #if 0 "
                            "because no local selector exists"
                        ),
                    },
                }
                try:
                    inputs = _make_inputs(
                        workload,
                        operator_shape,
                        dtype,
                        device=device,
                        seed=args.seed + workload_index,
                        with_decay_bias=args.decay_bias,
                    )
                    state_pool = inputs["state_pool"]
                    r = inputs["r"]
                    assert isinstance(state_pool, torch.Tensor)
                    assert isinstance(r, torch.Tensor)
                    ticket = flashrwkv2.prepare_tmix_wkv7_recurrent_metadata(
                        inputs["cu_seqlens"],
                        inputs["state_indices"],
                        total_tokens=r.shape[0],
                        state_pool_size=state_pool.shape[0],
                    )
                    correctness = _run_public_correctness(
                        inputs,
                        limits=limits,
                    )
                    row = {
                        **base,
                        "state_pool_size": state_pool.shape[0],
                        "state_pool_bytes": state_pool.numel() * state_pool.element_size(),
                        "correctness": correctness,
                    }
                    if not correctness["passed"]:
                        row["failure"] = "correctness gate failed"
                    elif not args.correctness_only:
                        flat_inputs = _flatten_inputs(inputs)
                        samples = _measure(
                            flat_inputs,
                            inputs,
                            warmup=args.warmup,
                            samples=args.samples,
                            sample_iters=args.sample_iters,
                            ticket=ticket,
                        )
                        row["timing"] = _measure_row(samples, workload)
                        print(
                            "RESULT "
                            f"operator_shape={operator_shape.name} B={workload.batch_size} "
                            f"T={workload.uniform_t or 'ragged'} dtype={dtype_name} "
                            "boundary=in_place correctness=passed "
                            f"p10_ms={row['timing']['p10_ms']:.6f} "
                            f"p50_ms={row['timing']['p50_ms']:.6f} "
                            f"p90_ms={row['timing']['p90_ms']:.6f} "
                            f"tok_s_p50={row['timing']['tok_s_p50']:.3f}"
                        )
                    else:
                        print(
                            "RESULT "
                            f"operator_shape={operator_shape.name} B={workload.batch_size} "
                            f"T={workload.uniform_t or 'ragged'} dtype={dtype_name} "
                            "boundary=in_place correctness=passed"
                        )
                except (RuntimeError, torch.cuda.OutOfMemoryError) as error:
                    row = {**base, "failure": f"{type(error).__name__}: {error}"}
                    torch.cuda.empty_cache()
                    print(
                        "RESULT "
                        f"operator_shape={operator_shape.name} B={workload.batch_size} "
                        f"T={workload.uniform_t or 'ragged'} dtype={dtype_name} "
                        f"correctness=failed reason={type(error).__name__}"
                    )
                results.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=list) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
