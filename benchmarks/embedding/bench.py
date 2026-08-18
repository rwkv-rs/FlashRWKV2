# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import json

import torch
from _timing import measure_cuda

from flashrwkv2.embedding import infer_embedding_ln0_forward_varlen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--channels", type=int, default=4096)
    args = parser.parse_args()
    device = torch.device("cuda")
    embedding = torch.randn(
        args.tokens, args.channels, device=device, dtype=torch.bfloat16
    )
    weight = torch.ones(args.channels, device=device, dtype=torch.bfloat16)
    bias = torch.zeros_like(weight)

    timing = measure_cuda(
        lambda: infer_embedding_ln0_forward_varlen(embedding, weight, bias)
    )
    print(
        json.dumps(
            {
                "operator": "infer_embedding_ln0_forward_varlen",
                "profile": f"tokens={args.tokens}/channels={args.channels}",
                "steady_state": "stateless inputs are reused",
                **timing,
            }
        )
    )


if __name__ == "__main__":
    main()
