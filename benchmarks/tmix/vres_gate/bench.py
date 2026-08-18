# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import json

import torch
from _timing import measure_cuda

from flashrwkv2.tmix.vres_gate import pretrain_tmix_vres_gate_bf16


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seqlen", type=int, default=32)
    parser.add_argument("--channels", type=int, default=4096)
    args = parser.parse_args()
    shape = (args.batch, args.seqlen, args.channels)
    value = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    first = torch.randn_like(value)
    v0 = torch.randn(args.channels, device="cuda", dtype=torch.bfloat16)
    v12 = torch.randn_like(value)
    timing = measure_cuda(
        lambda: pretrain_tmix_vres_gate_bf16(value, first, v0, v12)
    )
    print(
        json.dumps(
            {
                "operator": "pretrain_tmix_vres_gate_bf16",
                "profile": (
                    f"batch={args.batch}/seqlen={args.seqlen}/channels={args.channels}"
                ),
                "steady_state": "stateless inputs are reused",
                **timing,
            }
        )
    )


if __name__ == "__main__":
    main()
