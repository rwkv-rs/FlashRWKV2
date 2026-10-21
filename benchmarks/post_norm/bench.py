# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from _timing import measure_cuda

import flashrwkv2
from flashrwkv2.post_norm import infer_post_norm_output_forward_varlen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--channels", type=int, default=4096)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--timing-mode", choices=("eager", "graph"), default="eager")
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument(
        "--implementation", choices=("tilelang", "cuda-baseline"), default="tilelang"
    )
    args = parser.parse_args()
    if args.timing_mode == "graph":
        torch.cuda.set_stream(torch.cuda.Stream())
    tilelang = args.implementation == "tilelang"
    package_path = Path(flashrwkv2.__file__).resolve()
    if (package_path.parent / "_internal" / "linear.py").is_file() != tilelang:
        raise ValueError("--implementation must match the imported flashrwkv2 package")
    if args.tune and not tilelang:
        raise ValueError("--tune requires the TileLang implementation")
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16 if tilelang else torch.float16
    x = (
        torch.randn(args.tokens, args.channels, device=device)
        .bfloat16()
        .half()
        .to(dtype)
    )
    res = (
        torch.randn(args.tokens, args.channels, device=device)
        .bfloat16()
        .half()
        .to(dtype)
    )
    weight = torch.ones(args.channels, device=device, dtype=dtype)
    bias = torch.zeros_like(weight)
    tuning = None
    if args.tune:
        from flashrwkv2.post_norm.tilelang import tune_norm

        tuning = tune_norm(x, weight, bias, res=res)
    timing = measure_cuda(
        lambda: infer_post_norm_output_forward_varlen(x, res, weight, bias),
        timing_mode=args.timing_mode,
    )
    print(
        json.dumps(
            {
                "operator": "infer_post_norm_output_forward_varlen",
                "profile": f"tokens={args.tokens}/channels={args.channels}",
                "seed": args.seed,
                "implementation": args.implementation,
                "package_path": str(package_path),
                "input_dtype": str(dtype).removeprefix("torch."),
                "output_dtype": str(dtype).removeprefix("torch."),
                "tuning": tuning,
                "steady_state": "stateless inputs are reused",
                **timing,
            }
        )
    )


if __name__ == "__main__":
    main()
