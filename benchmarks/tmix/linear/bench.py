# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import json
import math
import subprocess
from collections.abc import Callable
from pathlib import Path

import torch

from flashrwkv2.tmix.linear import (
    infer_tmix_linear_attention_c2c_forward_varlen,
    infer_tmix_linear_rank_in_forward_varlen,
    infer_tmix_linear_rank_out_forward_varlen,
    infer_tmix_linear_rank_out_sigmoid_forward_varlen,
    infer_tmix_linear_rank_out_tanh_forward_varlen,
    infer_tmix_lowrank_in_forward_varlen,
    infer_tmix_lowrank_out_forward_varlen,
    infer_tmix_lowrank_vres_forward_varlen,
    infer_tmix_lowrank_wagv_in_forward_varlen,
)
from flashrwkv2.tmix.vres_gate import infer_tmix_vres_gate_forward_varlen

SOURCE_REVISION = "ee3308f6922e59f2166c7fac3c5a192340a2b48e"
DEFAULT_ROWS = (1, 4, 5, 7, 8, 9, 16, 24, 32, 48, 64, 96, 128, 192, 256, 512, 1024)
DEFAULT_RANKS = (96, 128, 480)


def _flashrwkv2_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[3],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return "unknown"
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=Path(__file__).resolve().parents[3],
        check=False,
        capture_output=True,
        text=True,
    )
    suffix = "+dirty" if dirty.returncode == 0 and dirty.stdout else ""
    return result.stdout.strip() + suffix


def _parse_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.replace(",", " ").split())
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError(
            "expected a non-empty list of positive integers"
        )
    return result


def _percentile(values: list[float], quantile: float) -> float:
    return float(
        torch.quantile(torch.tensor(values, dtype=torch.float64), quantile).item()
    )


def _layouts(
    runtime: torch.Tensor, layout: str
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    original = runtime.t().contiguous()
    if layout == "original":
        return original, None
    if layout == "runtime":
        return None, runtime
    return original, runtime


def _metrics(
    actual: tuple[torch.Tensor, ...], expected: tuple[torch.Tensor, ...]
) -> dict[str, float]:
    differences = [
        left.float() - right.float()
        for left, right in zip(actual, expected, strict=True)
    ]
    max_abs = max(float(difference.abs().max().item()) for difference in differences)
    square_sum = sum(
        float(difference.square().sum().item()) for difference in differences
    )
    elements = sum(difference.numel() for difference in differences)
    return {"max_abs": max_abs, "rmse": math.sqrt(square_sum / elements)}


def _time(
    call: Callable[[], tuple[torch.Tensor, ...]], warmup: int, samples: int
) -> list[float]:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    values = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        end.synchronize()
        values.append(float(start.elapsed_time(end)) * 1000.0)
    return values


def _case(
    operator: str,
    rows: int,
    channels: int,
    rank: int,
    layout: str,
    warmup: int,
    samples: int,
) -> dict[str, object]:
    device = torch.device("cuda")
    inputs = [
        torch.randn(rows, channels, device=device, dtype=torch.float16)
        for _ in range(4)
    ]
    rank_in_runtime = [
        torch.randn(channels, rank, device=device, dtype=torch.float16)
        for _ in range(4)
    ]
    rank_in_layouts = [_layouts(weight, layout) for weight in rank_in_runtime]
    rank_out_runtime = [
        torch.randn(rank, channels, device=device, dtype=torch.float16)
        for _ in range(4)
    ]
    rank_out_layouts = [_layouts(weight, layout) for weight in rank_out_runtime]
    value = torch.randn(rows, channels, device=device, dtype=torch.float16)
    value_first = torch.randn_like(value)
    value_bias = torch.randn(channels, device=device, dtype=torch.float16)

    individual_call: Callable[[], tuple[torch.Tensor, ...]]
    if operator == "lora":
        inputs[0].mul_(0.25)
        base_weight = torch.randn(
            channels, channels, device=device, dtype=torch.float16
        ) * 0.02
        adapter_a = rank_in_runtime[0].t().contiguous().mul_(0.02)
        adapter_b = rank_out_runtime[0].t().contiguous().mul_(0.02)
        scale = 0.25
        call = lambda: (
            infer_tmix_linear_attention_c2c_forward_varlen(
                inputs[0],
                base_weight,
                lora_a=adapter_a,
                lora_b=adapter_b,
                lora_scale=scale,
            ),
        )

        def composed_lora_call() -> tuple[torch.Tensor, ...]:
            base = infer_tmix_linear_attention_c2c_forward_varlen(
                inputs[0], base_weight
            )
            rank_features = infer_tmix_linear_rank_in_forward_varlen(
                inputs[0], weight_t=adapter_a
            )
            delta = infer_tmix_linear_rank_out_forward_varlen(
                rank_features, weight_t=adapter_b
            )
            return (base + delta * scale,)

        expected = (
            inputs[0].float() @ base_weight.float().t()
            + scale
            * (
                (inputs[0].float() @ adapter_a.float().t())
                @ adapter_b.float().t()
            ),
        )
        individual_call = composed_lora_call
        dispatch = "attention-c2c+rank-auto+direct-accumulate"
    elif operator == "rank-in":
        call = lambda: (
            infer_tmix_linear_rank_in_forward_varlen(
                inputs[0], weight=rank_in_layouts[0][1], weight_t=rank_in_layouts[0][0]
            ),
        )
        expected = (inputs[0].float() @ rank_in_runtime[0].float(),)
        individual_call = call
        dispatch = "linear_t_f16" if rows <= 7 and layout != "runtime" else "large-auto"
    elif operator == "rank-out":
        rank_input = torch.randn(rows, rank, device=device, dtype=torch.float16)
        call = lambda: (
            infer_tmix_linear_rank_out_forward_varlen(
                rank_input,
                weight=rank_out_layouts[0][1],
                weight_t=rank_out_layouts[0][0],
            ),
        )
        expected = (rank_input.float() @ rank_out_runtime[0].float(),)
        individual_call = call
        dispatch = "linear_t_f16" if rows <= 4 and layout != "runtime" else "large-auto"
    else:
        rank_inputs = lambda: infer_tmix_lowrank_wagv_in_forward_varlen(
            *inputs,
            *(weights[0] for weights in rank_in_layouts),
            w1_runtime=rank_in_layouts[0][1],
            a1_runtime=rank_in_layouts[1][1],
            g1_runtime=rank_in_layouts[2][1],
            v1_runtime=rank_in_layouts[3][1],
        )
        individual_rank_inputs = lambda: tuple(
            infer_tmix_linear_rank_in_forward_varlen(
                source, weight=weights[1], weight_t=weights[0]
            )
            for source, weights in zip(inputs, rank_in_layouts, strict=True)
        )

        def individual_rank_outputs(
            projected: tuple[torch.Tensor, ...],
        ) -> tuple[torch.Tensor, ...]:
            return (
                infer_tmix_linear_rank_out_tanh_forward_varlen(
                    projected[0],
                    weight=rank_out_layouts[0][1],
                    weight_t=rank_out_layouts[0][0],
                ),
                infer_tmix_linear_rank_out_forward_varlen(
                    projected[1],
                    weight=rank_out_layouts[1][1],
                    weight_t=rank_out_layouts[1][0],
                ),
                infer_tmix_linear_rank_out_sigmoid_forward_varlen(
                    projected[2],
                    weight=rank_out_layouts[2][1],
                    weight_t=rank_out_layouts[2][0],
                ),
            )

        def public_vres(
            projected: tuple[torch.Tensor, ...],
        ) -> tuple[torch.Tensor, ...]:
            return infer_tmix_lowrank_vres_forward_varlen(
                *projected,
                *(weights[0] for weights in rank_out_layouts),
                value,
                value_first,
                value_bias,
                w2_runtime=rank_out_layouts[0][1],
                a2_runtime=rank_out_layouts[1][1],
                g2_runtime=rank_out_layouts[2][1],
                v2_runtime=rank_out_layouts[3][1],
            )

        def individual_vres(
            projected: tuple[torch.Tensor, ...],
        ) -> tuple[torch.Tensor, ...]:
            outputs = individual_rank_outputs(projected)
            value_delta = infer_tmix_linear_rank_out_forward_varlen(
                projected[3],
                weight=rank_out_layouts[3][1],
                weight_t=rank_out_layouts[3][0],
            )
            return outputs + (
                infer_tmix_vres_gate_forward_varlen(
                    value, value_first, value_bias, value_delta
                ),
            )

        if operator == "wag":
            call = lambda: infer_tmix_lowrank_in_forward_varlen(
                *inputs[:3],
                *(weights[0] for weights in rank_in_layouts[:3]),
                w1_runtime=rank_in_layouts[0][1],
                a1_runtime=rank_in_layouts[1][1],
                g1_runtime=rank_in_layouts[2][1],
            )
            expected = tuple(
                source.float() @ weight.float()
                for source, weight in zip(inputs[:3], rank_in_runtime[:3], strict=True)
            )
            individual_call = lambda: individual_rank_inputs()[:3]
        elif operator == "wagv":
            call = rank_inputs
            expected = tuple(
                source.float() @ weight.float()
                for source, weight in zip(inputs, rank_in_runtime, strict=True)
            )
            individual_call = individual_rank_inputs
        else:
            projected = rank_inputs()
            if operator == "rank-out-group":
                call = lambda: infer_tmix_lowrank_out_forward_varlen(
                    *projected[:3],
                    *(weights[0] for weights in rank_out_layouts[:3]),
                    w2_runtime=rank_out_layouts[0][1],
                    a2_runtime=rank_out_layouts[1][1],
                    g2_runtime=rank_out_layouts[2][1],
                )
                expected = (
                    torch.tanh(projected[0].float()) @ rank_out_runtime[0].float(),
                    projected[1].float() @ rank_out_runtime[1].float(),
                    torch.sigmoid(projected[2].float()) @ rank_out_runtime[2].float(),
                )
                individual_call = lambda: individual_rank_outputs(projected)
            elif operator == "vres":
                call = lambda: public_vres(projected)
                individual_call = lambda: individual_vres(projected)
                value_delta = projected[3].float() @ rank_out_runtime[3].float()
                expected = (
                    torch.tanh(projected[0].float()) @ rank_out_runtime[0].float(),
                    projected[1].float() @ rank_out_runtime[1].float(),
                    torch.sigmoid(projected[2].float()) @ rank_out_runtime[2].float(),
                    value.float()
                    + (value_first.float() - value.float())
                    * torch.sigmoid(value_bias.float() + value_delta),
                )
            else:
                call = lambda: public_vres(rank_inputs())
                individual_call = lambda: individual_vres(individual_rank_inputs())
                rank_in_reference = tuple(
                    source.float() @ weight.float()
                    for source, weight in zip(inputs, rank_in_runtime, strict=True)
                )
                if not all(
                    torch.allclose(actual_rank.float(), reference, atol=0.08, rtol=0.08)
                    for actual_rank, reference in zip(
                        projected, rank_in_reference, strict=True
                    )
                ):
                    raise RuntimeError(
                        "projection-group rank-in correctness gate failed"
                    )
                projected_expected = tuple(
                    rank_feature.float() for rank_feature in projected
                )
                value_delta = projected_expected[3] @ rank_out_runtime[3].float()
                expected = (
                    torch.tanh(projected_expected[0]) @ rank_out_runtime[0].float(),
                    projected_expected[1] @ rank_out_runtime[1].float(),
                    torch.sigmoid(projected_expected[2]) @ rank_out_runtime[2].float(),
                    value.float()
                    + (value_first.float() - value.float())
                    * torch.sigmoid(value_bias.float() + value_delta),
                )
        if operator == "projection-group":
            dispatch = {
                "rank_in": (
                    "fused-composite"
                    if rows <= 7 and layout != "runtime"
                    else "large-auto"
                ),
                "rank_out": (
                    "fused-composite"
                    if rows <= 4 and layout != "runtime"
                    else "large-auto"
                ),
                "value_residual": (
                    "fused-composite"
                    if rows <= 4 and layout != "runtime"
                    else "large-auto+vres-gate"
                ),
            }
        else:
            limit = 4 if operator in ("rank-out-group", "vres") else 7
            dispatch = (
                "fused-composite"
                if rows <= limit and layout != "runtime"
                else "large-auto"
            )

    torch.cuda.reset_peak_memory_stats()
    actual = tuple(call())
    correctness = _metrics(actual, tuple(expected))
    if not all(torch.isfinite(tensor).all().item() for tensor in actual):
        raise RuntimeError("non-finite output")
    tolerance = 0.02 if operator == "lora" else 0.08
    if not all(
        torch.allclose(
            left.float(), right.float(), atol=tolerance, rtol=tolerance
        )
        for left, right in zip(actual, expected, strict=True)
    ):
        raise RuntimeError(f"correctness gate failed: {correctness}")
    individual_actual = tuple(individual_call())
    comparison_tolerance = 0.02 if operator == "lora" else 0.04
    if not all(
        torch.allclose(
            left.float(),
            right.float(),
            atol=comparison_tolerance,
            rtol=comparison_tolerance,
        )
        for left, right in zip(actual, individual_actual, strict=True)
    ):
        raise RuntimeError("public composite and individual canonical paths disagree")
    torch.cuda.reset_peak_memory_stats()
    raw_latency_us = _time(call, warmup, samples)
    public_peak_allocated = torch.cuda.max_memory_allocated()
    public_peak_reserved = torch.cuda.max_memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    individual_latency_us = _time(individual_call, warmup, samples)
    individual_peak_allocated = torch.cuda.max_memory_allocated()
    individual_peak_reserved = torch.cuda.max_memory_reserved()
    return {
        "operator": operator,
        "rows": rows,
        "channels": channels,
        "rank": rank,
        "weight_layout": layout,
        "expected_dispatch_family": dispatch,
        "warmup": warmup,
        "samples": samples,
        "raw_latency_us": raw_latency_us,
        "p10_us": _percentile(raw_latency_us, 0.10),
        "p50_us": _percentile(raw_latency_us, 0.50),
        "p90_us": _percentile(raw_latency_us, 0.90),
        "individual_raw_latency_us": individual_latency_us,
        "individual_p10_us": _percentile(individual_latency_us, 0.10),
        "individual_p50_us": _percentile(individual_latency_us, 0.50),
        "individual_p90_us": _percentile(individual_latency_us, 0.90),
        "correctness": correctness,
        "peak_allocated_bytes": public_peak_allocated,
        "peak_reserved_bytes": public_peak_reserved,
        "individual_peak_allocated_bytes": individual_peak_allocated,
        "individual_peak_reserved_bytes": individual_peak_reserved,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--operator",
        choices=(
            "rank-in",
            "rank-out",
            "lora",
            "wag",
            "wagv",
            "rank-out-group",
            "vres",
            "projection-group",
        ),
        default="wag",
    )
    parser.add_argument("--rows", type=_parse_ints, default=DEFAULT_ROWS)
    parser.add_argument("--ranks", type=_parse_ints, default=DEFAULT_RANKS)
    parser.add_argument("--channels", type=int, default=4096)
    parser.add_argument(
        "--layout", choices=("original", "runtime", "both"), default="both"
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=30)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.channels <= 0 or args.warmup < 0 or args.samples <= 0:
        raise ValueError(
            "channels and samples must be positive; warmup must be non-negative"
        )

    metadata = {
        "benchmark": "flashrwkv2_tmix_lowrank_dispatch",
        "source_revision": SOURCE_REVISION,
        "flashrwkv2_revision": _flashrwkv2_revision(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
    }
    for rows in args.rows:
        for rank in args.ranks:
            result = _case(
                args.operator,
                rows,
                args.channels,
                rank,
                args.layout,
                args.warmup,
                args.samples,
            )
            print(json.dumps(metadata | result), flush=True)


if __name__ == "__main__":
    main()
