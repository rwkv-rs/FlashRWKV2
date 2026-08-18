# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import json

import torch
from _timing import measure_cuda

from flashrwkv2.tmix.a_gate import pretrain_tmix_a_gate_bf16


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seqlen", type=int, default=32)
    parser.add_argument("--channels", type=int, default=4096)
    args = parser.parse_args()
    device = torch.device("cuda")
    a0 = torch.zeros(args.channels, device=device, dtype=torch.bfloat16)
    a12 = torch.zeros(
        args.batch,
        args.seqlen,
        args.channels,
        device=device,
        dtype=torch.bfloat16,
    )
    timing = measure_cuda(lambda: pretrain_tmix_a_gate_bf16(a0, a12))
    print(
        json.dumps(
            {
                "operator": "pretrain_tmix_a_gate_bf16",
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
