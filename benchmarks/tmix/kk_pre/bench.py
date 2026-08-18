# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import json

import torch
from _timing import measure_cuda

from flashrwkv2.tmix.kk_pre import pretrain_tmix_kk_pre_bf16


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--channels", type=int, default=4096)
    args = parser.parse_args()
    device = torch.device("cuda")
    key = torch.randn(
        args.batch,
        args.tokens,
        args.channels,
        device=device,
        dtype=torch.bfloat16,
    )
    key_scale = torch.randn(args.channels, device=device, dtype=torch.bfloat16)
    learning_rate = torch.randn_like(key)
    learning_rate_scale = torch.randn(
        args.channels, device=device, dtype=torch.bfloat16
    )
    timing = measure_cuda(
        lambda: pretrain_tmix_kk_pre_bf16(
            key, key_scale, learning_rate, learning_rate_scale
        )
    )
    print(
        json.dumps(
            {
                "operator": "pretrain_tmix_kk_pre_bf16",
                "profile": (
                    f"batch={args.batch}/tokens={args.tokens}/channels={args.channels}"
                ),
                "steady_state": "stateless inputs are reused",
                **timing,
            }
        )
    )


if __name__ == "__main__":
    main()
