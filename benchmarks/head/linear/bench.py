# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import json

import torch
from _timing import measure_cuda

from flashrwkv2.head.linear import infer_head_linear_all_forward_varlen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--channels", type=int, default=4096)
    parser.add_argument("--vocab", type=int, default=65536)
    args = parser.parse_args()
    device = torch.device("cuda")
    x = torch.randn(args.tokens, args.channels, device=device, dtype=torch.float16)
    weight = torch.randn(
        args.vocab, args.channels, device=device, dtype=torch.float16
    )
    timing = measure_cuda(lambda: infer_head_linear_all_forward_varlen(x, weight))
    print(
        json.dumps(
            {
                "operator": "infer_head_linear_all_forward_varlen",
                "profile": (
                    f"tokens={args.tokens}/channels={args.channels}/vocab={args.vocab}"
                ),
                "steady_state": "stateless inputs are reused",
                **timing,
            }
        )
    )


if __name__ == "__main__":
    main()
