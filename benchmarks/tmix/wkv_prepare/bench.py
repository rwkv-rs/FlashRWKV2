# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import json
import time

import torch

from flashrwkv2.tmix.wkv_prepare import infer_tmix_wkv_prepare_forward_varlen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--channels", type=int, default=4096)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=30)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    options = {"device": device, "dtype": torch.float16}
    shifted = [torch.randn(args.tokens, args.channels, **options) for _ in range(6)]
    dense = [torch.randn(args.channels, args.channels, **options) * 0.01 for _ in range(3)]
    rank_in = [torch.randn(args.rank, args.channels, **options) * 0.01 for _ in range(4)]
    rank_out = [torch.randn(args.channels, args.rank, **options) * 0.01 for _ in range(4)]
    runtime_in = [weight.t().contiguous() for weight in rank_in]
    runtime_out = [weight.t().contiguous() for weight in rank_out]
    vectors = [torch.zeros(args.channels, **options) for _ in range(4)]

    def run() -> tuple[torch.Tensor, ...]:
        return infer_tmix_wkv_prepare_forward_varlen(
            *shifted,
            *dense,
            *rank_in,
            *rank_out,
            *vectors,
            w1_runtime=runtime_in[0],
            a1_runtime=runtime_in[1],
            g1_runtime=runtime_in[2],
            v1_runtime=runtime_in[3],
            w2_runtime=runtime_out[0],
            a2_runtime=runtime_out[1],
            g2_runtime=runtime_out[2],
            v2_runtime=runtime_out[3],
            head_size=64,
            batch_size=1,
            max_seqlen=args.tokens,
        )

    for _ in range(args.warmup):
        outputs = run()
    torch.cuda.synchronize()
    latency = []
    for _ in range(args.samples):
        start = time.perf_counter()
        outputs = run()
        torch.cuda.synchronize()
        latency.append((time.perf_counter() - start) * 1e6)
    correctness = "passed" if all(torch.isfinite(output).all() for output in outputs) else "failed"
    print(
        json.dumps(
            {
                "operator": "infer_tmix_wkv_prepare_forward_varlen",
                "raw_latency_us": latency,
                "correctness": correctness,
            }
        )
    )


if __name__ == "__main__":
    main()
