# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from _timing import measure_cuda

import flashrwkv2
from flashrwkv2.tmix.tokenshift import (
    infer_tmix_postnorm_tokenshift_forward_varlen,
)
from flashrwkv2.tmix.wkv7 import prepare_tmix_wkv7_recurrent_metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seqlen", type=int, default=1)
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
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.batch <= 0 or args.seqlen <= 0 or args.channels <= 0:
        raise ValueError("batch, seqlen and channels must be positive")

    device = torch.device("cuda")
    rows = args.batch * args.seqlen
    dtype = torch.bfloat16 if tilelang else torch.float16

    def values(*shape, scale=1.0):
        return (torch.randn(shape, device=device) * scale).bfloat16().half().to(dtype)

    x = values(rows, args.channels)
    res = values(rows, args.channels)
    weight = torch.ones(args.channels, device=device, dtype=dtype)
    bias = torch.zeros_like(weight)
    parameters = [values(args.channels, scale=0.1) for _ in range(6)]
    shift_state = torch.zeros(args.batch, args.channels, device=device, dtype=dtype)
    cu_seqlens = torch.arange(
        0, rows + 1, args.seqlen, device=device, dtype=torch.int32
    )
    state_indices = torch.arange(args.batch, device=device, dtype=torch.int32)
    ticket = prepare_tmix_wkv7_recurrent_metadata(
        cu_seqlens,
        state_indices,
        total_tokens=rows,
        state_pool_size=args.batch,
        max_seqlen=args.seqlen,
    )
    tuning = {}
    if args.tune:
        from flashrwkv2.post_norm.tilelang import norm, tune_norm
        from flashrwkv2.tmix.tokenshift.tilelang import (
            tune_fused_tokenshift,
            tune_tokenshift,
        )

        if args.seqlen == 1:
            tuning["fused"] = tune_fused_tokenshift(
                x,
                res,
                weight,
                bias,
                parameters,
                shift_state,
                ticket._state_indices_snapshot(),
                ticket._status(),
            )
        else:
            tuning["norm"] = tune_norm(x, weight, bias, res=res)
            normalized = norm(x, weight, bias, res=res)
            tuning["tokenshift"] = tune_tokenshift(
                normalized,
                parameters,
                ticket._query_start_loc_snapshot(),
                shift_state,
                _status=ticket._status(),
            )

    def run() -> tuple[torch.Tensor, ...]:
        return infer_tmix_postnorm_tokenshift_forward_varlen(
            x,
            res,
            weight,
            bias,
            *parameters,
            shift_state_pool=shift_state,
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
            max_seqlen=args.seqlen,
            validated_metadata=ticket,
        )

    timing = measure_cuda(
        run, before_batch=shift_state.zero_, timing_mode=args.timing_mode
    )
    print(
        json.dumps(
            {
                "operator": "infer_tmix_postnorm_tokenshift_forward_varlen",
                "batch": args.batch,
                "seqlen": args.seqlen,
                "channels": args.channels,
                "seed": args.seed,
                "implementation": args.implementation,
                "package_path": str(package_path),
                "input_dtype": str(dtype).removeprefix("torch."),
                "shift_state_dtype": str(shift_state.dtype).removeprefix("torch."),
                "tuning": tuning,
                "profile": (
                    f"batch={args.batch}/seqlen={args.seqlen}/channels={args.channels}"
                ),
                "steady_state": "shift state is reset before each timing batch; metadata preparation is untimed",
                **timing,
            }
        )
    )


if __name__ == "__main__":
    main()
