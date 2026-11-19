# SPDX-License-Identifier: MIT
"""Time a full writeback and traversal cycle against materialized FP16 decode."""

import argparse
import json
from pathlib import Path

import torch
from _timing import measure_cuda

from flashrwkv2 import (
    infer_tmix_wkv7_recurrent_fp16_forward_varlen,
    prepare_tmix_wkv7_recurrent_fp16_state_from_tensor,
    prepare_tmix_wkv7_recurrent_metadata,
)
from flashrwkv2.tmix.wkv7.tilelang import _infer_recurrent


def compare(heads, batch, verify_only, slot_order):
    initial = torch.randn(batch, heads, 64, 64, device="cuda").mul_(0.05).half()
    pool = initial.clone()
    state = prepare_tmix_wkv7_recurrent_fp16_state_from_tensor(pool)
    cycle_steps = 6 if state._delta[0].shape[2] == 6 else 2
    steps = [
        torch.randn(6, batch, heads, 64, device="cuda").mul_(0.04).bfloat16()
        for _ in range(cycle_steps)
    ]
    cu = torch.arange(batch + 1, device="cuda", dtype=torch.int32)
    slots = [cu[:-1].flip((0,)) if slot_order == "reverse" and i % 2 else cu[:-1]
             for i in range(cycle_steps)]
    tickets = [
        prepare_tmix_wkv7_recurrent_metadata(
            cu,
            indices,
            total_tokens=batch,
            state_pool_size=batch,
            max_seqlen=1,
        )
        for indices in slots
    ]
    inputs = [step.unbind() for step in steps]

    def dense_step(i):
        return _infer_recurrent(
            *inputs[i], pool, cu, slots[i], tickets[i]._status(),
            block_values=64, prefetch_tokens=1, row_order=tickets[i]._row_order(),
        )

    def serving_step(i):
        return infer_tmix_wkv7_recurrent_fp16_forward_varlen(
            *inputs[i], state=state, cu_seqlens=cu, state_indices=slots[i],
            max_seqlen=1, validated_metadata=tickets[i],
        )

    def ordinary():
        return tuple(dense_step(i) for i in range(cycle_steps))

    def serving():
        return tuple(serving_step(i) for i in range(cycle_steps))

    def reset():
        state.zero_()
        pool.copy_(initial)

    first = dense_step(0)
    mid = pool.clone()
    reference = (first, *(dense_step(i) for i in range(1, cycle_steps)), pool.clone())

    def check(outputs):
        for actual, expected in zip((*outputs, pool), reference, strict=True):
            assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))

    reset()
    outputs = serving()
    state.materialize_slots_(slots[0])
    check(outputs)

    # A prefix-cache reader requests half the slots after the first token.
    reset()
    first = serving_step(0)
    deferred = torch.equal(pool.view(torch.int16), initial.view(torch.int16))
    state.materialize_slots_(slots[0][::2].contiguous())
    assert torch.equal(pool[::2].view(torch.int16), mid[::2].view(torch.int16))
    outputs = (first, *(serving_step(i) for i in range(1, cycle_steps)))
    state.materialize_slots_(slots[0])
    check(outputs)

    times = (
        {}
        if verify_only
        else {
            name: measure_cuda(
                call,
                timing_mode="graph",
                warmup=10,
                batches=7,
                launches_per_batch=192,
                before_batch=reset,
            )
            for name, call in (("ordinary", ordinary), ("serving", serving))
        }
    )
    return {
        "heads": heads,
        "batch": batch,
        "bitwise_equal": True,
        "first_step_deferred": deferred,
        "tokens_per_cycle": cycle_steps,
        "log_bytes_per_slot": state.memory_layout["private_bytes_per_slot"],
        **times,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--slot-order", choices=("fixed", "reverse"), default="fixed")
    args = parser.parse_args()
    torch.cuda.set_stream(torch.cuda.Stream())
    torch.manual_seed(938)
    results = [
        compare(heads, batch, args.verify_only, args.slot_order)
        for heads in (40, 64)
        for batch in (16, 64, 256)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(),
                "slot_order": args.slot_order,
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
