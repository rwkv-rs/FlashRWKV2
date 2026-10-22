# SPDX-License-Identifier: MIT
"""Measure explicit kernel candidates only after exact-output validation."""

import hashlib
import json
import math
import os
from functools import cache
from pathlib import Path
from statistics import median
from tempfile import NamedTemporaryFile

import torch


@cache
def _profile_root():
    """Share verified execution profiles across processes using the same kernels."""
    import tilelang

    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256(repr((tilelang.__version__, torch.version.cuda)).encode())
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return Path(torch.hub.get_dir()).parent / "flashrwkv2" / "tuning" / digest.hexdigest()


def _profile_path(owner, encoded_key):
    name = hashlib.sha256(encoded_key.encode()).hexdigest() + ".json"
    return _profile_root() / owner / name


@cache
def _read_profile(owner, encoded_key):
    try:
        record = json.loads(_profile_path(owner, encoded_key).read_text())
    except FileNotFoundError:
        return None
    if not isinstance(record, dict):
        raise ValueError("invalid verified kernel tuning profile")
    config = record.get("config")
    values = config if isinstance(config, list) else [config]
    if (
        record.get("key") != json.loads(encoded_key)
        or record.get("bitwise_equal") is not True
        or not values
        or any(type(value) is not int or value <= 0 for value in values)
    ):
        raise ValueError("invalid verified kernel tuning profile")
    return tuple(config) if isinstance(config, list) else config


def get_config(configs, owner, key, default=None):
    if key not in configs:
        encoded = json.dumps(key, default=str, separators=(",", ":"))
        config = _read_profile(owner, encoded)
        configs[key] = default if config is None else config
    return configs.get(key, default)


def save_config(owner, key, config):
    """Publish a selection only after the owner's exact-output checks and timing."""
    encoded = json.dumps(key, default=str, separators=(",", ":"))
    path = _profile_path(owner, encoded)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w", dir=path.parent, suffix=".tmp", delete=False
    ) as output:
        json.dump(
            {"key": json.loads(encoded), "config": config, "bitwise_equal": True},
            output,
        )
    os.replace(output.name, path)
    _read_profile.cache_clear()


def compile_candidates(keys, cache, spec):
    """Compile candidates sharing compiler options after serial TIR elaboration."""
    import tilelang

    pending, programs = [], []
    for key in dict.fromkeys(keys):
        if key in cache:
            continue
        try:
            program, options = spec(*key)
        except RuntimeError:
            # The owner records individual errors through its normal launcher.
            continue
        pending.append(key)
        programs.append(program)
    if programs:
        kernels = tilelang.par_compile(
            programs, **options, ignore_error=True
        )
        cache.update(
            (key, kernel)
            for key, kernel in zip(pending, kernels, strict=True)
            if kernel is not None
        )


def projection_finalists(measurements):
    """Retain three verified tiles per copy span for sustained graph timing."""
    spans = {}
    for item in measurements:
        if item.get("bitwise_equal"):
            spans.setdefault(item["config"][3], []).append(item)
    return tuple(
        item["config"]
        for items in spans.values()
        for item in sorted(items, key=lambda item: item["milliseconds"])[:3]
    )


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
        # Metadata tickets keep the stream on which the owner prepared them.
        capture_stream = torch.cuda.current_stream()
        if capture_stream == torch.cuda.default_stream():
            capture_stream = None
        with torch.cuda.graph(graph, stream=capture_stream):
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
        replays = math.ceil(10.0 / start.elapsed_time(end))
        for _ in range(replays):
            graph.replay()
        samples = []
        for _ in range(3):
            start.record()
            for _ in range(replays):
                graph.replay()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) / (repeat * replays))
        record["raw_milliseconds"] = samples
        record["milliseconds"] = median(samples)
        record["launches_per_graph"] = repeat
    best = min(eligible, key=lambda item: item[1]["milliseconds"])[0]
    return best, measurements
