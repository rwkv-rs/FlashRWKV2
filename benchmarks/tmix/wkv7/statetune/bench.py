# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from _timing import measure_cuda

from flashrwkv2.tmix.wkv7.statetune import statetune_tmix_wkv7_recurrent_fp32io16


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument(
        "--output", type=Path, default=Path("/tmp/flashrwkv2-statetune.json")
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    state = torch.randn(1, args.heads, 64, 64, device=device, dtype=torch.float32)
    values = [
        torch.randn(
            args.tokens, args.heads, 64, device=device, dtype=torch.float16
        )
        for _ in range(6)
    ]
    offsets = torch.tensor([0, 1], device=device, dtype=torch.int32)
    starts = torch.tensor([0], device=device, dtype=torch.int32)
    ends = torch.tensor([args.tokens], device=device, dtype=torch.int32)
    initial_state = state.clone()

    def run() -> None:
        statetune_tmix_wkv7_recurrent_fp32io16(state, offsets, starts, ends, *values)

    timing = measure_cuda(run, before_batch=lambda: state.copy_(initial_state))
    result = {
        "operator": "statetune_tmix_wkv7_recurrent_fp32io16",
        "timing": timing,
        "steady_state": "state reset before each timing batch",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
