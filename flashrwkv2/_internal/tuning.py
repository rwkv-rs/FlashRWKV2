# SPDX-License-Identifier: MIT
"""Measure explicit kernel candidates only after exact-output validation."""

import torch


@torch.no_grad()
def select_config(candidates, run, *, check=None, reference=None, warmup=3, repeat=20):
    """Return (configuration, measurements), timing CUDA Graph replay.

    ``check`` may return higher-precision outputs than ``run`` to verify
    accumulators before timing the production output dtype. The owning module
    retains its shape/device cache and validates invariance across shapes.
    """
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("kernel tuning must finish before CUDA Graph capture")
    candidates = tuple(candidates)
    check = run if check is None else check
    reference = check(candidates[0]) if reference is None else reference
    expected = (reference,) if isinstance(reference, torch.Tensor) else tuple(reference)
    measurements, eligible = [], []
    for config in candidates:
        actual = check(config)
        values = (actual,) if isinstance(actual, torch.Tensor) else tuple(actual)
        equal = len(values) == len(expected) and all(
            torch.equal(value, target) for value, target in zip(values, expected)
        )
        record = {"config": config, "bitwise_equal": equal}
        measurements.append(record)
        if equal:
            eligible.append((config, record))
    if not eligible:
        raise RuntimeError("no kernel configuration satisfies the bitwise contract")
    for config, record in eligible:
        for _ in range(warmup):
            run(config)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(repeat):
                run(config)
        # Instantiate the graph and events before measuring steady device time.
        graph.replay()
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        end.record()
        end.synchronize()
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        record["milliseconds"] = start.elapsed_time(end) / repeat
    best = min(eligible, key=lambda item: item[1]["milliseconds"])[0]
    return best, measurements
