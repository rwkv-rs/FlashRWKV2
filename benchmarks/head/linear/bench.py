# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from _timing import measure_cuda

import flashrwkv2
from flashrwkv2.head.linear import (
    infer_head_linear_all_forward_varlen,
    pretrain_head_linear_bf16,
)


def benchmark(args):
    tilelang = args.implementation == "tilelang"
    source_uses_tilelang = (
        Path(flashrwkv2.__file__).parent / "_internal" / "linear.py"
    ).is_file()
    if tilelang != source_uses_tilelang:
        raise ValueError("--implementation must match the imported flashrwkv2 package")
    if args.tune and not tilelang:
        raise ValueError("--tune requires the TileLang implementation")
    torch.manual_seed(args.seed)
    backward = args.backward or bool(args.adapter_rank)
    if args.compare_torch and (not tilelang or backward):
        raise ValueError(
            "--compare-torch requires TileLang base inference without adapters"
        )
    dtype = torch.bfloat16 if tilelang or backward else torch.float16

    def values(shape, dtype, scale):
        # BF16 significands and varied exponents exercise FP32 rounding while
        # both input formats receive identical logical values.
        return (torch.randn(shape, device="cuda") * scale).bfloat16().half().to(dtype)

    x = values((args.tokens, args.channels), dtype, 1.0).requires_grad_(backward)
    weight = values((args.vocab, args.channels), dtype, 0.03).requires_grad_(
        backward and not args.adapter_rank
    )
    adapters = {}
    inputs = (x, weight)
    if args.adapter_rank:
        a = (
            torch.randn(args.adapter_rank, args.channels, device="cuda") * 0.01
        ).requires_grad_()
        b = (
            torch.randn(args.vocab, args.adapter_rank, device="cuda") * 0.01
        ).requires_grad_()
        adapters = {"lora_a": a, "lora_b": b, "use_nora": True}
        inputs = (x, a, b)
    tuning = {}
    if args.tune:
        from flashrwkv2._internal.linear import _NoRA, linear, tune_linear

        tuning["base"] = tune_linear(x, weight, torch.float32)
        if adapters:
            normalized = _NoRA.apply(a.detach())
            tuning["adapter_a"] = tune_linear(x, normalized)
            features = linear(x.detach(), normalized)
            tuning["adapter_b"] = tune_linear(
                features, b.detach().bfloat16(), torch.float32
            )
    # Record the production precision of each backend; the input values match.
    output_dtype = (
        torch.float32 if tilelang else torch.bfloat16 if backward else torch.float16
    )
    grad = values((args.tokens, args.vocab), output_dtype, 1.0) if backward else None

    def run():
        if backward:
            output = pretrain_head_linear_bf16(x, weight, **adapters)
            torch.autograd.grad(output, inputs, grad)
        else:
            output = infer_head_linear_all_forward_varlen(x, weight)
        return output

    sample = run()
    output_dtype = sample.dtype
    torch_comparison = None
    if args.compare_torch:
        # Require FP32 accumulation on the native side, matching the declared
        # contract rather than allowing BF16 partial-reduction truncation.
        previous_reduction = (
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction_split_k,
        )
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (False, False)
        try:

            def native(values=x):
                return torch.mm(values, weight.t(), out_dtype=torch.float32)

            reference = native()
            checks = []
            for row in sorted({0, x.shape[0] // 2, x.shape[0] - 1}):
                actual = native(x[row : row + 1])
                expected = reference[row : row + 1]
                checks.append(
                    {
                        "mode": f"row={row}",
                        "fp32_bitwise_equal": torch.equal(actual, expected),
                        "bf16_bitwise_equal": torch.equal(
                            actual.bfloat16(), expected.bfloat16()
                        ),
                        "max_abs_error": float((actual - expected).abs().max()),
                    }
                )
            chunked = torch.cat([native(chunk) for chunk in x.split(7)])
            checks.append(
                {
                    "mode": "chunks=7",
                    "fp32_bitwise_equal": torch.equal(chunked, reference),
                    "bf16_bitwise_equal": torch.equal(
                        chunked.bfloat16(), reference.bfloat16()
                    ),
                    "max_abs_error": float((chunked - reference).abs().max()),
                }
            )
            # M=1 cannot exercise a different GEMM shape by slicing itself.
            repeated = native(x[:1].expand(17, -1).contiguous())[:1]
            checks.append(
                {
                    "mode": "repeat_first_row=17",
                    "fp32_bitwise_equal": torch.equal(repeated, reference[:1]),
                    "bf16_bitwise_equal": torch.equal(
                        repeated.bfloat16(), reference[:1].bfloat16()
                    ),
                    "max_abs_error": float((repeated - reference[:1]).abs().max()),
                }
            )
            torch_comparison = {
                "operator": "torch.mm",
                "preferred_blas_library": str(
                    torch.backends.cuda.preferred_blas_library()
                ),
                "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                "CUBLASLT_WORKSPACE_SIZE": os.environ.get("CUBLASLT_WORKSPACE_SIZE"),
                "input_dtype": str(x.dtype).removeprefix("torch."),
                "output_dtype": str(reference.dtype).removeprefix("torch."),
                "allow_bf16_reduced_precision_reduction": False,
                "allow_bf16_reduced_precision_reduction_split_k": False,
                "batch_invariance": checks,
                "tilelang_max_abs_difference": float((sample - reference).abs().max()),
                "timing": measure_cuda(native, timing_mode=args.timing_mode),
            }
        finally:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
                previous_reduction
            )
    del sample
    return {
        "operator": "pretrain_head_linear_bf16"
        if backward
        else "infer_head_linear_all_forward_varlen",
        "profile": f"tokens={args.tokens}/channels={args.channels}/vocab={args.vocab}/adapter_rank={args.adapter_rank}",
        "tokens": args.tokens,
        "channels": args.channels,
        "vocab": args.vocab,
        "implementation": args.implementation,
        "package_path": str(Path(flashrwkv2.__file__).resolve()),
        "seed": args.seed,
        "input_distribution": "FP32 normal samples rounded through BF16 and FP16; weight scale=0.03, FP32 adapter scale=0.01",
        "input_dtype": str(dtype).removeprefix("torch."),
        "output_dtype": str(output_dtype).removeprefix("torch."),
        "upstream_dtype": str(grad.dtype).removeprefix("torch.") if backward else None,
        "accumulation_dtype": "float32",
        "forward_backward": backward,
        "tuning": tuning,
        "torch_comparison": torch_comparison,
        "steady_state": "stateless inputs are reused",
        **measure_cuda(run, timing_mode=args.timing_mode),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--channels", type=int, default=4096)
    parser.add_argument("--vocab", type=int, default=65536)
    parser.add_argument(
        "--adapter-rank", type=int, default=0, help="enable NoRA forward+backward"
    )
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument(
        "--implementation", choices=("tilelang", "cuda-baseline"), default="tilelang"
    )
    parser.add_argument("--timing-mode", choices=("eager", "graph"), default="eager")
    parser.add_argument(
        "--compare-torch",
        action="store_true",
        help="compare native BF16-to-FP32 mm latency and batch/chunk bitwise invariance",
    )
    parser.add_argument(
        "--torch-blas",
        choices=("default", "cublaslt"),
        default="default",
        help="select the native comparison configuration before CUDA initialization",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="verify and measure output tiles before timing",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="emit JSONL for six model widths and M=1/27/256",
    )
    args = parser.parse_args()
    if args.torch_blas == "cublaslt":
        if not args.compare_torch:
            parser.error("--torch-blas cublaslt requires --compare-torch")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        os.environ["CUBLASLT_WORKSPACE_SIZE"] = "1"
        torch.backends.cuda.preferred_blas_library("cublaslt")
    cases = [(args.tokens, args.channels)]
    if args.sweep:
        cases = [
            (rows, channels)
            for channels in (768, 1024, 2048, 2560, 4096, 5120)
            for rows in (1, 27, 256)
        ]
    for tokens, channels in cases:
        case = argparse.Namespace(
            **{**vars(args), "tokens": tokens, "channels": channels}
        )
        print(json.dumps(benchmark(case)))


if __name__ == "__main__":
    main()
