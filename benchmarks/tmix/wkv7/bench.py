# SPDX-License-Identifier: MIT

"""Benchmark the raw RWKV-7 recurrent inference operators.

This benchmark owns only operator shapes and packed workloads.  It does not
define a model, compose an operator graph, or multiply any model-level
quantity into the measured latency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from _timing import measure_cuda

import flashrwkv2


@dataclass(frozen=True, slots=True)
class ImportedSourceFamily:
    name: str
    repository: str
    revision: str
    license: str
    paths: tuple[str, ...]


SOURCE_FAMILY = ImportedSourceFamily(
    name="albatross-faster3a-2607-recurrent",
    repository="https://github.com/BlinkDL/Albatross",
    revision="3e41bc43ed5e8332927ddd7e0ce4816cf200a6ea",
    license="Apache-2.0",
    paths=(
        "faster3a_2607/cuda/rwkv7_wkv_fp32_v2.cu",
        "faster3a_2607/cuda/rwkv7_wkv_fp16_v2.cu",
        "faster3a_2607/cuda/rwkv7_wkv_deltalog_v3a.cu",
        "csrc/sm120/tmix/wkv7/infer_recurrent_fp32io16_forward_varlen.cu",
        "csrc/sm120/tmix/wkv7/infer_recurrent_fp32io16_forward_varlen.cpp",
        "csrc/sm120/tmix/wkv7/infer_recurrent_fp16_forward_varlen.cu",
        "csrc/sm120/tmix/wkv7/infer_recurrent_fp16_forward_varlen.cpp",
        "csrc/sm120/tmix/wkv7/infer_recurrent_deltalog_fp16_forward_varlen.cu",
        "csrc/sm120/tmix/wkv7/infer_recurrent_deltalog_fp16_forward_varlen.cpp",
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


KERNEL_SPECS = {
    "fp32io16": KernelSpec(
        provider="Albatross",
        name="infer_tmix_wkv7_recurrent_fp32io16_forward_varlen",
        layouts=("packed",),
        state_dtype="float32",
        token_dtypes=("float16", "bfloat16"),
        head_sizes=(64, 128, 256),
        source_family=SOURCE_FAMILY.name,
    ),
    "fp16": KernelSpec(
        provider="Albatross",
        name="infer_tmix_wkv7_recurrent_fp16_forward_varlen",
        layouts=("packed", "state_indices", "kv_state"),
        state_dtype="float16",
        token_dtypes=("float16",),
        head_sizes=(64, 128, 256),
        source_family=SOURCE_FAMILY.name,
    ),
}


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
    "decode_b16_t1": (1,) * 16,
    "decode_b64_t1": (1,) * 64,
    "decode_b320_t1": (1,) * 320,
    "decode_b960_t1": (1,) * 960,
    "decode_heavy_b960": (1,) * 768 + (2,) * 144 + (4,) * 32 + (16,) * 16,
    "mixed_b320": (1,) * 256 + (8,) * 48 + (64,) * 16,
    "balanced_b320": tuple(range(1, 17)) * 20,
    "prefill_skew_b64": (1,) * 48 + (16,) * 8 + (128,) * 4 + (512,) * 4,
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
        "csrc/sm120/tmix/wkv7/infer_recurrent_fp16_forward_varlen.cu",
        "csrc/sm120/tmix/wkv7/infer_recurrent_fp16_forward_varlen.cpp",
        "csrc/sm120/tmix/wkv7/infer_recurrent_deltalog_fp16_forward_varlen.cu",
        "csrc/sm120/tmix/wkv7/infer_recurrent_deltalog_fp16_forward_varlen.cpp",
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


def _make_inputs(
    workload: Workload,
    operator_shape: OperatorShape,
    dtype: torch.dtype,
    *,
    device: torch.device,
    seed: int,
    with_decay_bias: bool,
    operator: str,
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
    state_dtype = torch.float32 if operator == "fp32io16" else torch.float16
    state_pool = torch.randn(
        (
            state_slots,
            operator_shape.num_heads,
            operator_shape.head_size,
            operator_shape.head_size,
        ),
        device=device,
        dtype=state_dtype,
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
        "elapsed_state_pool": torch.zeros(
            state_slots, device=device, dtype=torch.int32
        ),
        "decay_bias": decay_bias,
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
    operator: str,
    max_seqlen: int,
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
    if operator == "fp32io16":
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
    else:
        elapsed = inputs["elapsed_state_pool"]
        assert isinstance(elapsed, torch.Tensor)
        extension.tmix_wkv7_recurrent_fp16_from_decay_logits(
            cu_seqlens,
            state_indices,
            elapsed,
            state,
            *flat_inputs,
            output,
            scale,
            decay_bias=decay_bias,
            validated_metadata=ticket,
            max_seqlen=max_seqlen,
        )


def _measure(
    flat_inputs: tuple[torch.Tensor, ...],
    inputs: dict[str, torch.Tensor | None],
    *,
    workload: Workload,
    ticket: object,
    operator: str,
) -> dict[str, object]:
    reset_state = inputs["state_pool"]
    if not isinstance(reset_state, torch.Tensor):
        raise TypeError("benchmark state pool is incomplete")
    output = torch.empty_like(flat_inputs[3])
    state = reset_state.clone()
    reset_elapsed = inputs["elapsed_state_pool"]
    if not isinstance(reset_elapsed, torch.Tensor):
        raise TypeError("benchmark elapsed state is incomplete")
    elapsed = reset_elapsed.clone()
    launch_inputs = {**inputs, "elapsed_state_pool": elapsed}

    def reset() -> None:
        state.copy_(reset_state)
        elapsed.copy_(reset_elapsed)

    def run() -> None:
        _timed_native_launch(
            flat_inputs,
            launch_inputs,
            output=output,
            state=state,
            ticket=ticket,
            operator=operator,
            max_seqlen=max(workload.lengths),
        )

    timing = measure_cuda(run, before_batch=reset)
    timing["tokens_per_second_mean"] = (
        workload.total_tokens * 1_000_000.0 / float(timing["mean_us"])
    )
    return timing


def _selected_fp32_family(
    workload: Workload, dtype: torch.dtype, num_heads: int
) -> str:
    """Report the measured family selected for canonical [K,V] state."""

    uniform_t = workload.uniform_t
    if dtype == torch.float16 and uniform_t is not None and (
        (uniform_t == 1 and workload.batch_size <= 96)
        or (uniform_t == 2 and workload.batch_size <= 21)
        or (uniform_t == 3 and workload.batch_size <= 3)
        or (uniform_t == 4 and workload.batch_size in {1, 3})
        or (workload.batch_size == 1 and 5 <= uniform_t <= 11)
    ):
        return "wkv_fp32_v2_small_warp_kv8"
    if workload.batch_size == 1 and workload.uniform_t == 1:
        return "wkv_fp32_v2_kv_tile"
    sequence_heads = workload.batch_size * num_heads
    max_seqlen = max(workload.lengths)
    if dtype == torch.float16:
        if max_seqlen == 5 and sequence_heads <= 80:
            return "wkv_fp32_v2_kv_tloop_group4_vt8"
        if 6 <= max_seqlen <= 9 and sequence_heads <= 80:
            tile = 4 if sequence_heads <= 40 else 8
            return f"wkv_fp32_v2_kv_tloop_group4_vt{tile}"
        if max_seqlen >= 10 and sequence_heads <= 48:
            return "wkv_fp32_v2_kv_tloop_group4_vt4"
    return "wkv_fp32_v2"


def _selected_fp16_family(workload: Workload, num_heads: int) -> str:
    sequence_heads = workload.batch_size * num_heads
    if workload.uniform_t == 1 and sequence_heads >= 20000:
        return "wkv_fp16_kv_vector"
    if (
        workload.uniform_t == 1
        and 15000 <= sequence_heads < 20000
    ) or (max(workload.lengths) > 1 and sequence_heads >= 1280):
        return "wkv_fp16_kv_warp_pair"
    return "wkv_fp16_existing_family"


DELTALOG_EXPERIMENTS = (
    (8, 2, False),
    (64, 3, False),
    (256, 3, False),
    (512, 4, False),
    (256, 3, True),
)


def _measure_deltalog_experiment(
    *,
    batch_size: int,
    merge_interval: int,
    staggered: bool,
    device: torch.device,
    seed: int,
) -> dict[str, object]:
    shape = OPERATOR_SHAPES["h64d64"]
    workload = Workload(f"{batch_size}x1", (1,) * batch_size)
    inputs = _make_inputs(
        workload,
        shape,
        torch.float16,
        device=device,
        seed=seed,
        with_decay_bias=True,
        operator="fp16",
    )
    packed = _flatten_inputs(inputs)
    state_template = inputs["state_pool"]
    elapsed_template = inputs["elapsed_state_pool"]
    cu_seqlens = inputs["cu_seqlens"]
    state_indices = inputs["state_indices"]
    decay_bias = inputs["decay_bias"]
    assert isinstance(state_template, torch.Tensor)
    assert isinstance(elapsed_template, torch.Tensor)
    assert isinstance(cu_seqlens, torch.Tensor)
    assert isinstance(state_indices, torch.Tensor)
    ticket = flashrwkv2.prepare_tmix_wkv7_recurrent_metadata(
        cu_seqlens,
        state_indices,
        total_tokens=batch_size,
        state_pool_size=state_template.shape[0],
        max_seqlen=1,
    )
    extension = flashrwkv2._C
    if extension is None:
        raise RuntimeError("flashrwkv2._C is not loaded")
    output = torch.empty_like(packed[3])

    normal_state = state_template.clone()
    normal_elapsed = elapsed_template.clone()

    def reset_normal() -> None:
        normal_state.copy_(state_template)
        normal_elapsed.copy_(elapsed_template)

    def run_normal_cycle() -> None:
        for _ in range(merge_interval):
            extension.tmix_wkv7_recurrent_fp16_from_decay_logits(
                cu_seqlens,
                state_indices,
                normal_elapsed,
                normal_state,
                *packed,
                output,
                1.0,
                decay_bias=decay_bias,
                validated_metadata=ticket,
                max_seqlen=1,
            )

    normal_timing = measure_cuda(run_normal_cycle, before_batch=reset_normal)

    deltalog_state = state_template.clone()
    deltalog_elapsed = elapsed_template.clone()
    phase_template = torch.zeros_like(elapsed_template)
    if staggered:
        phase_template[:batch_size] = torch.arange(
            batch_size, device=device, dtype=torch.int32
        ).remainder_(merge_interval)
    phase = phase_template.clone()
    logs_template = torch.zeros(
        (
            merge_interval - 1,
            5,
            state_template.shape[0],
            shape.num_heads,
            64,
        ),
        device=device,
        dtype=torch.float16,
    )
    logs = logs_template.clone()

    def reset_deltalog() -> None:
        deltalog_state.copy_(state_template)
        deltalog_elapsed.copy_(elapsed_template)
        phase.copy_(phase_template)
        logs.copy_(logs_template)

    def run_deltalog_cycle() -> None:
        for _ in range(merge_interval):
            extension.tmix_wkv7_recurrent_deltalog_fp16_from_decay_logits(
                cu_seqlens,
                state_indices,
                deltalog_elapsed,
                phase,
                deltalog_state,
                logs,
                *packed,
                output,
                1.0,
                decay_bias=decay_bias,
                validated_metadata=ticket,
            )

    deltalog_timing = measure_cuda(
        run_deltalog_cycle, before_batch=reset_deltalog
    )
    normal_mean = float(normal_timing["mean_us"])
    deltalog_mean = float(deltalog_timing["mean_us"])
    return {
        "name": (
            f"c4096_b{batch_size}_m{merge_interval}"
            + ("_staggered" if staggered else "")
        ),
        "B": batch_size,
        "H": shape.num_heads,
        "D": 64,
        "M": merge_interval,
        "staggered_phase": staggered,
        "boundary": "complete_cycle_returns_each_slot_to_its_initial_phase",
        "normal_fp16": normal_timing,
        "slot_native_deltalog": deltalog_timing,
        "deltalog_speedup": normal_mean / deltalog_mean,
        "deltalog_lower_latency": deltalog_mean < normal_mean,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--operator",
        choices=("fp32io16", "fp16"),
        default="fp32io16",
    )
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
    parser.add_argument("--seed", type=int, default=20260804)
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
    root = _repo_root()
    workloads = (
        tuple(_parse_case(value) for value in args.cases)
        if args.cases is not None
        else default_workloads(args.stress)
    )
    family = SOURCE_FAMILY
    spec = KERNEL_SPECS[args.operator]
    git = _git_metadata(root)
    extension = getattr(flashrwkv2, "_C", None)
    if extension is None:
        raise RuntimeError("flashrwkv2._C is not loaded; build the CUDA extension first")
    extension_path = Path(getattr(extension, "__file__", ""))
    payload: dict[str, object] = {
        "schema_version": 1,
        "benchmark": f"flashrwkv2_wkv7_recurrent_{args.operator}",
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
            "state_dtype": "float32" if args.operator == "fp32io16" else "float16",
            "operator": args.operator,
            "stress": args.stress,
            "decay_bias": args.decay_bias,
            "seed": args.seed,
        },
        "results": [],
        "experiments": [],
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
                    "state_dtype": (
                        "float32" if args.operator == "fp32io16" else "float16"
                    ),
                    "boundary": "preallocated_native_consecutive_in_place_launches",
                    "steady_state": "state pool is reset before each timing batch",
                    "selected_kernel_family": (
                        _selected_fp32_family(
                            workload, dtype, operator_shape.num_heads
                        )
                        if args.operator == "fp32io16"
                        else _selected_fp16_family(workload, operator_shape.num_heads)
                    ),
                    "dispatch": {
                        "batch_size": workload.batch_size,
                        "max_seqlen": max(workload.lengths),
                        "policy": (
                            "automatic packed-varlen policy; FP32IO16 includes "
                            "the D64 group4 t-loop selector and FP16 includes "
                            "native [K,V] warp-pair/vector thresholds"
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
                        operator=args.operator,
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
                    row = {
                        **base,
                        "state_pool_size": state_pool.shape[0],
                        "state_pool_bytes": state_pool.numel() * state_pool.element_size(),
                    }
                    flat_inputs = _flatten_inputs(inputs)
                    row["timing"] = _measure(
                        flat_inputs,
                        inputs,
                        workload=workload,
                        ticket=ticket,
                        operator=args.operator,
                    )
                    print(
                        "RESULT "
                        f"operator_shape={operator_shape.name} B={workload.batch_size} "
                        f"T={workload.uniform_t or 'ragged'} dtype={dtype_name} "
                        "boundary=in_place "
                        f"mean_us={row['timing']['mean_us']:.6f} "
                        f"tok_s_mean={row['timing']['tokens_per_second_mean']:.3f}"
                    )
                except (RuntimeError, torch.cuda.OutOfMemoryError) as error:
                    row = {**base, "failure": f"{type(error).__name__}: {error}"}
                    torch.cuda.empty_cache()
                    print(
                        "RESULT "
                        f"operator_shape={operator_shape.name} B={workload.batch_size} "
                        f"T={workload.uniform_t or 'ragged'} dtype={dtype_name} "
                        f"timing=unavailable reason={type(error).__name__}"
                    )
                results.append(row)

    if (
        args.operator == "fp16"
        and os.environ.get("FLASHRWKV_BENCHMARK_STATUS") == "candidate"
        and hasattr(
            extension,
            "tmix_wkv7_recurrent_deltalog_fp16_from_decay_logits",
        )
    ):
        experiments = payload["experiments"]
        assert isinstance(experiments, list)
        for index, (batch_size, merge_interval, staggered) in enumerate(
            DELTALOG_EXPERIMENTS
        ):
            experiment = _measure_deltalog_experiment(
                batch_size=batch_size,
                merge_interval=merge_interval,
                staggered=staggered,
                device=device,
                seed=args.seed + 1000 + index,
            )
            experiments.append(experiment)
            print(
                "EXPERIMENT "
                f"name={experiment['name']} "
                f"normal_mean_us={experiment['normal_fp16']['mean_us']:.6f} "
                "deltalog_mean_us="
                f"{experiment['slot_native_deltalog']['mean_us']:.6f}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=list) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
