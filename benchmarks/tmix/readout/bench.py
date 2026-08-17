# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import json
import time

import torch

from flashrwkv2.tmix.readout import infer_tmix_readout_forward_varlen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--channels", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=30)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    options = {"device": device, "dtype": torch.float16}
    packed = [torch.randn(args.tokens, args.channels, **options) for _ in range(5)]
    affine = [torch.randn(args.channels, **options) for _ in range(3)]
    output_weight = torch.randn(args.channels, args.channels, **options) * 0.01

    def run() -> torch.Tensor:
        return infer_tmix_readout_forward_varlen(
            packed[0],
            packed[1],
            packed[2],
            packed[3],
            affine[0],
            affine[1],
            affine[2],
            packed[4],
            output_weight,
            head_size=64,
            batch_size=1,
            max_seqlen=args.tokens,
        )

    for _ in range(args.warmup):
        output = run()
    torch.cuda.synchronize()
    latency = []
    for _ in range(args.samples):
        start = time.perf_counter()
        output = run()
        torch.cuda.synchronize()
        latency.append((time.perf_counter() - start) * 1e6)
    print(
        json.dumps(
            {
                "operator": "infer_tmix_readout_forward_varlen",
                "raw_latency_us": latency,
                "correctness": "passed" if torch.isfinite(output).all() else "failed",
            }
        )
    )


if __name__ == "__main__":
    main()
