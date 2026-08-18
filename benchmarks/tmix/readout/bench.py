# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import json

import torch
from _timing import measure_cuda

from flashrwkv2.tmix.readout import infer_tmix_readout_forward_varlen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--channels", type=int, default=4096)
    args = parser.parse_args()
    options = {"device": torch.device("cuda"), "dtype": torch.float16}
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

    timing = measure_cuda(run)
    print(
        json.dumps(
            {
                "operator": "infer_tmix_readout_forward_varlen",
                "profile": f"tokens={args.tokens}/channels={args.channels}",
                "steady_state": "stateless inputs are reused",
                **timing,
            }
        )
    )


if __name__ == "__main__":
    main()
