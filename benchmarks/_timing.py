"""Private, uniform CUDA benchmark timing helpers."""

from __future__ import annotations

from collections.abc import Callable
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
) -> dict[str, Any]:
    """Measure device time for 10,000 launches without per-launch sync noise."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if warmup < 0 or batches <= 0 or launches_per_batch <= 0:
        raise ValueError("timing counts must be positive and warmup non-negative")
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()

    batch_means_us: list[float] = []
    for _ in range(batches):
        if before_batch is not None:
            before_batch()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(launches_per_batch):
            call()
        end.record()
        end.synchronize()
        batch_means_us.append(
            float(start.elapsed_time(end) * 1000.0 / launches_per_batch)
        )
    total_launches = batches * launches_per_batch
    return {
        "warmup_launches": warmup,
        "batches": batches,
        "launches_per_batch": launches_per_batch,
        "total_launches": total_launches,
        "raw_batch_mean_us": batch_means_us,
        "mean_us": sum(batch_means_us) / len(batch_means_us),
    }
