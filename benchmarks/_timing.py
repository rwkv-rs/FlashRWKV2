"""Private, uniform CUDA benchmark timing helpers."""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter
from typing import Any

import torch

DEFAULT_WARMUP = 100
DEFAULT_BATCHES = 10
DEFAULT_LAUNCHES_PER_BATCH = 1000


def measure_cuda(
    call: Callable[[], Any],
    *,
    warmup: int = DEFAULT_WARMUP,
    batches: int = DEFAULT_BATCHES,
    launches_per_batch: int = DEFAULT_LAUNCHES_PER_BATCH,
    before_batch: Callable[[], Any] | None = None,
    timing_mode: str = "eager",
) -> dict[str, Any]:
    """Measure eager dispatch wall time or one captured batch's device time.

    Eager includes Python dispatch, allocation and synchronization. Graph mode
    captures the whole launch batch, so the timed replay has no Python loop
    between its operators. Warmup, capture and ``before_batch`` are untimed.
    """
    if warmup < 0 or batches <= 0 or launches_per_batch <= 0:
        raise ValueError("timing counts must be positive and warmup non-negative")
    if timing_mode not in ("eager", "graph"):
        raise ValueError("timing_mode must be eager or graph")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if before_batch is not None:
        before_batch()
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()

    graph = None
    if timing_mode == "graph":
        if before_batch is not None:
            before_batch()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(launches_per_batch):
                call()

    batch_means_us: list[float] = []
    for _ in range(batches):
        if before_batch is not None:
            before_batch()
        torch.cuda.synchronize()
        if graph is None:
            start = perf_counter()
            for _ in range(launches_per_batch):
                call()
            torch.cuda.synchronize()
            elapsed_us = (perf_counter() - start) * 1.0e6
        else:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            graph.replay()
            end_event.record()
            end_event.synchronize()
            elapsed_us = float(start_event.elapsed_time(end_event) * 1000.0)
        batch_means_us.append(elapsed_us / launches_per_batch)
    total_launches = batches * launches_per_batch
    return {
        "warmup_launches": warmup,
        "batches": batches,
        "launches_per_batch": launches_per_batch,
        "total_launches": total_launches,
        "raw_batch_mean_us": batch_means_us,
        "mean_us": sum(batch_means_us) / len(batch_means_us),
        "timing_mode": timing_mode,
        "timing_scope": (
            "wall time including eager Python dispatch and final synchronization"
            if graph is None
            else "CUDA event time for one graph replay containing a full launch batch"
        ),
    }
