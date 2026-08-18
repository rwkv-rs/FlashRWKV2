# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import json

import torch
from _timing import measure_cuda

from flashrwkv2.head.l2wrap_ce import pretrain_head_l2wrap_ce_bf16


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--channels", type=int, default=64)
    args = parser.parse_args()
    device = torch.device("cuda")
    hidden = torch.randn(
        args.batch,
        args.tokens,
        args.channels,
        device=device,
        dtype=torch.bfloat16,
    )
    weight = torch.randn(
        65536, args.channels, device=device, dtype=torch.bfloat16
    )
    targets = torch.randint(
        65536, (args.batch * args.tokens,), device=device, dtype=torch.int64
    )
    timing = measure_cuda(
        lambda: pretrain_head_l2wrap_ce_bf16(hidden, weight, targets)
    )
    print(
        json.dumps(
            {
                "operator": "pretrain_head_l2wrap_ce_bf16",
                "profile": (
                    f"batch={args.batch}/tokens={args.tokens}/channels={args.channels}"
                ),
                "steady_state": "forward outputs are discarded after each launch",
                **timing,
            }
        )
    )


if __name__ == "__main__":
    main()
