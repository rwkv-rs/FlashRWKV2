# SPDX-License-Identifier: MIT
"""Time an exact two-token FP16 DeltaLog cycle against public WKV decode.

The append leaves the bound state at the start of the cycle. The merge replays
that update, rounds to FP16, then applies and materializes the second update.
"""

import argparse
import json
from pathlib import Path

import tilelang
import tilelang.language as T
import torch
from _timing import measure_cuda

from flashrwkv2 import (
    infer_tmix_wkv7_recurrent_fp16_forward_varlen,
    prepare_tmix_wkv7_recurrent_fp16_state_from_tensor,
    prepare_tmix_wkv7_recurrent_metadata,
)
from flashrwkv2.tmix.wkv7.tilelang import (
    _COMPILE_FLAGS,
    _DECAY_EXP2,
    _LOG2E,
    _PASSES,
    _dot_keys,
    _update_state,
)


@tilelang.jit(pass_configs=_PASSES, compile_flags=_COMPILE_FLAGS)
def _deltalog_step(heads):
    batch, capacity = T.dynamic("batch, capacity")

    @T.prim_func
    def kernel(
        R: T.Tensor((batch, heads, 64), "bfloat16"),
        W: T.Tensor((batch, heads, 64), "bfloat16"),
        K: T.Tensor((batch, heads, 64), "bfloat16"),
        V: T.Tensor((batch, heads, 64), "bfloat16"),
        A: T.Tensor((batch, heads, 64), "bfloat16"),
        B: T.Tensor((batch, heads, 64), "bfloat16"),
        State: T.Tensor((capacity, heads, 64, 64), "float16"),
        Log: T.Tensor((capacity, heads, 5, 64), "float32"),
        Phase: T.Tensor((capacity, heads), "uint8"),
        Slots: T.Tensor((batch,), "int32"),
        Y: T.Tensor((batch, heads, 64), "bfloat16"),
    ):
        with T.Kernel(heads, batch, threads=64) as (head, seq):
            slot = Slots[seq]
            pending = T.alloc_shared((1,), "uint8")
            if T.get_thread_binding() == 0:
                pending[0] = Phase[slot, head]
            T.sync_threads()
            state = T.alloc_fragment((64, 64), "float32")
            partial = T.alloc_fragment((1, 64), "float32")
            state_dot_a = T.alloc_fragment((64,), "float32")
            output = T.alloc_fragment((64,), "float32")
            previous = T.alloc_shared((5, 64), "float32")
            r = T.alloc_shared((64,), "float32")
            decay = T.alloc_shared((64,), "float32")
            k = T.alloc_shared((64,), "float32")
            v = T.alloc_fragment((64,), "float32")
            a = T.alloc_shared((64,), "float32")
            b = T.alloc_shared((64,), "float32")
            T.annotate_layout(
                {
                    state: T.Fragment(
                        (64, 64),
                        forward_thread_fn=lambda key, col: col,
                        forward_index_fn=lambda key, col: key,
                    ),
                    partial: T.Fragment(
                        (1, 64),
                        forward_thread_fn=lambda group, col: col,
                        forward_index_fn=lambda group, col: 0,
                    ),
                }
            )
            for key, col in T.Parallel(64, 64):
                state[key, col] = State[slot, head, key, col]
            if pending[0] != 0:
                for row, key in T.Parallel(5, 64):
                    previous[row, key] = Log[slot, head, row, key]
                for key, col in T.Parallel(64, 64):
                    replay = T.fma(
                        previous[1, key],
                        previous[2, col],
                        T.fma(
                            previous[3, key],
                            previous[4, col],
                            previous[0, key] * state[key, col],
                        ),
                    )
                    state[key, col] = T.cast(T.cast(replay, "float16"), "float32")
            for key in T.Parallel(64):
                r[key] = R[seq, head, key]
                w = T.cast(W[seq, head, key], "float32")
                decay[key] = T.exp2(_DECAY_EXP2 / (1.0 + T.exp2(-_LOG2E * w)))
                k[key] = K[seq, head, key]
                v[key] = V[seq, head, key]
                a[key] = A[seq, head, key]
                b[key] = B[seq, head, key]
            _dot_keys(state, a, partial, state_dot_a)
            _update_state(state, decay, k, v, b, state_dot_a, 0, "float16")
            _dot_keys(state, r, partial, output)
            for col in T.Parallel(64):
                Y[seq, head, col] = T.cast(output[col], "bfloat16")
            if pending[0] != 0:
                for key, col in T.Parallel(64, 64):
                    State[slot, head, key, col] = T.cast(state[key, col], "float16")
            else:
                for key in T.Parallel(64):
                    Log[slot, head, 0, key] = decay[key]
                    Log[slot, head, 1, key] = k[key]
                    Log[slot, head, 2, key] = v[key]
                    Log[slot, head, 3, key] = b[key]
                    Log[slot, head, 4, key] = state_dot_a[key]

            if T.get_thread_binding() == 0:
                Phase[slot, head] = T.cast(1 - pending[0], "uint8")

    return kernel


@tilelang.jit(pass_configs=_PASSES, compile_flags=_COMPILE_FLAGS)
def _materialize(heads):
    batch, capacity = T.dynamic("batch, capacity")

    @T.prim_func
    def kernel(
        State: T.Tensor((capacity, heads, 64, 64), "float16"),
        Log: T.Tensor((capacity, heads, 5, 64), "float32"),
        Phase: T.Tensor((capacity, heads), "uint8"),
        Slots: T.Tensor((batch,), "int32"),
    ):
        with T.Kernel(heads, batch, threads=64) as (head, seq):
            slot = Slots[seq]
            pending = T.alloc_shared((1,), "uint8")
            if T.get_thread_binding() == 0:
                pending[0] = Phase[slot, head]
            T.sync_threads()
            if pending[0] != 0:
                previous = T.alloc_shared((5, 64), "float32")
                for row, key in T.Parallel(5, 64):
                    previous[row, key] = Log[slot, head, row, key]
                for key, col in T.Parallel(64, 64):
                    replay = T.fma(
                        previous[1, key],
                        previous[2, col],
                        T.fma(
                            previous[3, key],
                            previous[4, col],
                            previous[0, key]
                            * T.cast(State[slot, head, key, col], "float32"),
                        ),
                    )
                    State[slot, head, key, col] = T.cast(replay, "float16")
                if T.get_thread_binding() == 0:
                    Phase[slot, head] = 0

    return kernel


def compare(heads, batch, verify_only, slot_order):
    steps = [
        torch.randn(6, batch, heads, 64, device="cuda").mul_(0.04).bfloat16()
        for _ in range(2)
    ]
    initial = torch.randn(batch, heads, 64, 64, device="cuda").mul_(0.05).half()
    pool = initial.clone()
    state = prepare_tmix_wkv7_recurrent_fp16_state_from_tensor(pool)
    cu = torch.arange(batch + 1, device="cuda", dtype=torch.int32)
    slots = [cu[:-1], cu[:-1].flip((0,)) if slot_order == "reverse" else cu[:-1]]
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
        return infer_tmix_wkv7_recurrent_fp16_forward_varlen(
            *inputs[i],
            state=state,
            cu_seqlens=cu,
            state_indices=slots[i],
            max_seqlen=1,
            validated_metadata=tickets[i],
        )

    def ordinary():
        return dense_step(0), dense_step(1)

    log = torch.empty(batch, heads, 5, 64, device="cuda", dtype=torch.float32)
    phase = torch.zeros((batch, heads), device="cuda", dtype=torch.uint8)
    outputs = [torch.empty_like(step[0]) for step in steps]
    kernel, flush = _deltalog_step(heads), _materialize(heads)

    def delta_step(i):
        kernel(*inputs[i], pool, log, phase, slots[i], outputs[i])

    def delta():
        delta_step(0)
        delta_step(1)

    first = dense_step(0)
    mid = pool.clone()
    reference = first, dense_step(1), pool.clone()

    def check():
        for actual, expected in zip((*outputs, pool), reference, strict=True):
            assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
        assert not phase.any().item()

    pool.copy_(initial)
    delta()
    check()

    # A prefix-cache reader requests half the slots after the first token.
    pool.copy_(initial)
    delta_step(0)
    selected = slots[0][::2].contiguous()
    flush(pool, log, phase, selected)
    assert torch.equal(pool[::2].view(torch.int16), mid[::2].view(torch.int16))
    assert torch.equal(pool[1::2].view(torch.int16), initial[1::2].view(torch.int16))
    delta_step(1)
    flush(pool, log, phase, slots[1])
    check()

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
                before_batch=lambda: pool.copy_(initial),
            )
            for name, call in (("ordinary", ordinary), ("deltalog", delta))
        }
    )
    return {
        "heads": heads,
        "batch": batch,
        "bitwise_equal": True,
        "log_bytes_per_slot": log[0].numel() * log.element_size() + phase[0].numel(),
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
