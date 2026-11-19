# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import json

import torch
from _timing import measure_cuda

from flashrwkv2 import (
    infer_sampling_six_parameter_forward_varlen,
    setup_sampling_states,
)


def _profile(batch_size: int, vocab_size: int) -> dict[str, object]:
    generator = torch.Generator(device="cuda").manual_seed(20260807)
    logits = torch.randn(
        batch_size,
        vocab_size,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    num_slots = batch_size * 2
    slot_indices = torch.arange(0, num_slots, 2, dtype=torch.int32, device="cuda")
    states = setup_sampling_states(20260807, num_slots)
    initial_states = states.clone()
    top_k = 50
    top_p = 0.9
    penalties = torch.zeros(
        num_slots, vocab_size, dtype=torch.float32, device="cuda"
    )
    presence = torch.full((batch_size,), 0.1, device="cuda")
    frequency = torch.full((batch_size,), 0.1, device="cuda")
    decays = torch.full((batch_size,), 0.996, device="cuda")
    temperatures = torch.ones(batch_size, device="cuda")
    top_ks = torch.full((batch_size,), top_k, dtype=torch.int32, device="cuda")
    top_ps = torch.full((batch_size,), top_p, device="cuda")
    call = lambda: infer_sampling_six_parameter_forward_varlen(
        logits,
        penalties,
        states,
        slot_indices,
        presence_penalty=presence,
        frequency_penalty=frequency,
        penalty_decay=decays,
        temperature=temperatures,
        top_k=top_ks,
        top_p=top_ps,
    )

    def reset() -> None:
        states.copy_(initial_states)
        penalties.zero_()

    timing = measure_cuda(call, before_batch=reset)
    return {
        "profile": f"six_parameter/b{batch_size}/v{vocab_size}",
        "provider": "six_parameter",
        "batch_size": batch_size,
        "vocab_size": vocab_size,
        "steady_state": "sampling state and penalties reset before each timing batch",
        **timing,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 32, 128])
    parser.add_argument("--vocab-sizes", type=int, nargs="+", default=[65536, 131072])
    args = parser.parse_args()
    results = [
        _profile(batch_size, vocab_size)
        for vocab_size in args.vocab_sizes
        for batch_size in args.batch_sizes
    ]
    print(json.dumps({"operator": "sampling", "results": results}))


if __name__ == "__main__":
    main()
