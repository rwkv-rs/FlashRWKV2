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
def _deltalog_step(heads, merge):
    batch = T.dynamic("batch")

    @T.prim_func
    def kernel(
        X: T.Tensor((6, batch, heads, 64), "bfloat16"),
        State: T.Tensor((batch, heads, 64, 64), "float16"),
        Log: T.Tensor((batch, heads, 5, 64), "float32"),
        Y: T.Tensor((batch, heads, 64), "bfloat16"),
    ):
        with T.Kernel(heads, batch, threads=64) as (head, seq):
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
                state[key, col] = State[seq, head, key, col]
            if merge:
                for row, key in T.Parallel(5, 64):
                    previous[row, key] = Log[seq, head, row, key]
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
                r[key] = X[0, seq, head, key]
                w = T.cast(X[1, seq, head, key], "float32")
                decay[key] = T.exp2(_DECAY_EXP2 / (1.0 + T.exp2(-_LOG2E * w)))
                k[key] = X[2, seq, head, key]
                v[key] = X[3, seq, head, key]
                a[key] = X[4, seq, head, key]
                b[key] = X[5, seq, head, key]
            _dot_keys(state, a, partial, state_dot_a)
            _update_state(state, decay, k, v, b, state_dot_a, 0, "float16")
            _dot_keys(state, r, partial, output)
            for col in T.Parallel(64):
                Y[seq, head, col] = T.cast(output[col], "bfloat16")
            if merge:
                for key, col in T.Parallel(64, 64):
                    State[seq, head, key, col] = T.cast(state[key, col], "float16")
            else:
                for key in T.Parallel(64):
                    Log[seq, head, 0, key] = decay[key]
                    Log[seq, head, 1, key] = k[key]
                    Log[seq, head, 2, key] = v[key]
                    Log[seq, head, 3, key] = b[key]
                    Log[seq, head, 4, key] = state_dot_a[key]

    return kernel


def compare(heads, batch):
    steps = [
        torch.randn(6, batch, heads, 64, device="cuda").mul_(0.04).bfloat16()
        for _ in range(2)
    ]
    initial = torch.randn(batch, heads, 64, 64, device="cuda").mul_(0.05).half()
    pool = initial.clone()
    state = prepare_tmix_wkv7_recurrent_fp16_state_from_tensor(pool)
    cu = torch.arange(batch + 1, device="cuda", dtype=torch.int32)
    slots = cu[:-1]
    ticket = prepare_tmix_wkv7_recurrent_metadata(
        cu,
        slots,
        total_tokens=batch,
        state_pool_size=batch,
        max_seqlen=1,
    )
    inputs = [step.unbind() for step in steps]

    def ordinary():
        return tuple(
            infer_tmix_wkv7_recurrent_fp16_forward_varlen(
                *values,
                state=state,
                cu_seqlens=cu,
                state_indices=slots,
                max_seqlen=1,
                validated_metadata=ticket,
            )
            for values in inputs
        )

    log = torch.empty(batch, heads, 5, 64, device="cuda", dtype=torch.float32)
    outputs = [torch.empty_like(step[0]) for step in steps]
    append, merge = (_deltalog_step(heads, flag) for flag in (False, True))

    def delta():
        append(steps[0], pool, log, outputs[0])
        merge(steps[1], pool, log, outputs[1])

    reference = (*ordinary(), pool.clone())
    pool.copy_(initial)
    delta()
    for actual, expected in zip((*outputs, pool), reference, strict=True):
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))

    times = {
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
    return {
        "heads": heads,
        "batch": batch,
        "bitwise_equal": True,
        "log_bytes_per_slot": log[0].numel() * log.element_size(),
        **times,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.cuda.set_stream(torch.cuda.Stream())
    torch.manual_seed(938)
    results = [compare(heads, batch) for heads in (40, 64) for batch in (16, 64, 256)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"gpu": torch.cuda.get_device_name(), "results": results}, indent=2)
        + "\n"
    )


if __name__ == "__main__":
    main()
