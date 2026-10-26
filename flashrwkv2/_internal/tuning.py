# SPDX-License-Identifier: MIT
"""Persistent verified selections and explicit preparation before CUDA capture."""

import ast
import fcntl
import hashlib
import json
import math
import os
import subprocess
import warnings
from contextvars import ContextVar
from functools import cache, wraps
from pathlib import Path
from statistics import median
from tempfile import NamedTemporaryFile

import torch

_SCHEMA = 2
_SESSION = ContextVar("flashrwkv2_tuning", default=None)
_TUNING = ContextVar("flashrwkv2_tuning_active", default=False)
_REVALIDATE = ContextVar("flashrwkv2_revalidate", default=None)
_LOADED = {}
_WARNED = set()
_WARMED_DEFAULTS = set()
_OWNER_SOURCES = {
    "linear": "_internal/linear.py",
    "post_norm": "post_norm/tilelang.py",
    "tmix/readout": "tmix/readout/tilelang.py",
    "tmix/tokenshift": "tmix/tokenshift/tilelang.py",
    "tmix/tokenshift/fused": "tmix/tokenshift/tilelang.py",
    "tmix/wkv7": "tmix/wkv7/tilelang.py",
    "tmix/wkv_prepare": "tmix/wkv_prepare/tilelang.py",
}


@cache
def get_tuning_cache_dir():
    """Return the fixed cache root; only FLASHRWKV2_CACHE_DIR overrides it."""
    configured = os.environ.get("FLASHRWKV2_CACHE_DIR") or "~/.cache/flashrwkv2"
    return Path(configured).expanduser().resolve()


@cache
def _profile_root():
    return get_tuning_cache_dir() / "tuning" / f"v{_SCHEMA}"


@cache
def device_class(device):
    props = torch.cuda.get_device_properties(device)
    return (props.name, props.major, props.minor, props.multi_processor_count)


@cache
def _compiler_identity():
    from tilelang.contrib.cc import get_cplus_compiler
    from tilelang.env import CUDA_HOME

    nvcc = str(Path(CUDA_HOME) / "bin" / "nvcc") if CUDA_HOME else None
    versions = []
    for compiler in (nvcc, get_cplus_compiler()):
        versions.append(None if compiler is None else subprocess.run(
            [compiler, "--version"], check=True, capture_output=True, text=True, timeout=10,
        ).stdout)
    flags = {name: os.environ.get(name) for name in (
        "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS", "TL_CUTLASS_PATH", "TL_TEMPLATE_PATH",
    )}
    return versions, flags


@cache
def _fingerprint(owner):
    """Hash the owner's source and transitive local implementation imports.

    AST serialization ignores comments, documentation and installation paths.
    Public __init__ wrappers and the persistence implementation are not kernels.
    Compiler flags and shape dispatch in implementation modules remain inputs.
    """
    import tilelang

    root = Path(__file__).resolve().parents[1]
    pending = [_OWNER_SOURCES[owner]]
    sources = {}
    while pending:
        relative = pending.pop()
        if relative in sources:
            continue
        path = root / relative
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                    node.body.pop(0)
            if isinstance(node, ast.ImportFrom) and node.level:
                base = path.parent
                for _ in range(node.level - 1):
                    base = base.parent
                dependency = base.joinpath(*(node.module or "").split(".")).with_suffix(".py")
                if dependency.is_file() and dependency.name not in {"__init__.py", "tuning.py"}:
                    pending.append(dependency.relative_to(root).as_posix())
        sources[relative] = ast.dump(tree, include_attributes=False)
    payload = (_SCHEMA, tilelang.__version__, torch.__version__, torch.version.cuda, _compiler_identity(), sources)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _encode(key):
    return json.dumps(key, default=str, separators=(",", ":"))


def _profile_path(owner, encoded_key):
    name = hashlib.sha256(encoded_key.encode()).hexdigest() + ".json"
    return _profile_root() / owner / _fingerprint(owner) / name


def _validate_record(record, encoded_key, *, owner=None):
    if not isinstance(record, dict):
        raise ValueError("invalid verified kernel tuning profile")
    config = record.get("config")
    values = config if isinstance(config, list) else [config]
    if (
        record.get("key") != json.loads(encoded_key)
        or record.get("bitwise_equal") is not True
        or not values
        or any(type(value) is not int or value <= 0 for value in values)
        or (owner is not None and (
            record.get("schema") != _SCHEMA
            or record.get("fingerprint") != _fingerprint(owner)
            or record.get("owner") != owner
        ))
    ):
        raise ValueError("invalid verified kernel tuning profile")
    return tuple(config) if isinstance(config, list) else config


def _read_profile(owner, encoded_key):
    try:
        record = json.loads(_profile_path(owner, encoded_key).read_text())
    except FileNotFoundError:
        return None
    return _validate_record(record, encoded_key, owner=owner)


def _capturing():
    return torch.cuda.is_initialized() and torch.cuda.is_current_stream_capturing()


def _legacy_config(directory, owner, key):
    # Legacy keys used the product name; preserve every remaining shape/dtype field.
    legacy_key = (key[0][0], *key[1:])
    encoded = _encode(legacy_key)
    name = hashlib.sha256(encoded.encode()).hexdigest() + ".json"
    path = Path(directory) / owner / name
    try:
        return _validate_record(json.loads(path.read_text()), encoded)
    except FileNotFoundError:
        return None


def get_config(configs, owner, key, default=None, *, tune=None):
    # Owner benchmarks deliberately inject candidates into their process cache.
    if _TUNING.get():
        return configs.get(key, default)
    session = _SESSION.get()
    identity = (str(_profile_root()), owner, key)
    # Explicit preparation reloads each key once, including files published or
    # removed by another process after an earlier eager call.
    refresh = session is not None and (owner, key) not in session["records"]
    config = None if refresh else _LOADED.get(identity)
    status = "hit"
    if config is None:
        if _capturing() and identity in _WARMED_DEFAULTS and session is None:
            return default
        if _capturing():
            raise RuntimeError(f"unprepared tuning key for {owner}; run prepare_tuning before CUDA Graph capture")
        encoded = _encode(key)
        config = _read_profile(owner, encoded)
        if config is None and session is not None and session["mode"] == "tune":
            if tune is None:
                raise RuntimeError(f"no tuning callback for {owner}")
            path = _profile_path(owner, encoded)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.with_suffix(".lock").open("a+b") as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                config = _read_profile(owner, encoded)
                if config is None:
                    legacy = session["legacy_dir"]
                    candidate = None if legacy is None else _legacy_config(legacy, owner, key)
                    active = _TUNING.set(True)
                    validation = _REVALIDATE.set(candidate)
                    try:
                        tune()
                    finally:
                        _REVALIDATE.reset(validation)
                        _TUNING.reset(active)
                    config = _read_profile(owner, encoded)
                    if config is None:
                        raise RuntimeError(f"tuner did not publish {owner}")
                    status = "revalidated" if candidate is not None else "tuned"
    if config is not None:
        _LOADED[identity] = config
        _WARMED_DEFAULTS.discard(identity)
        configs[key] = config
    else:
        status = "missing"
        # Never cache defaults: a later publication must become visible.
        _LOADED.pop(identity, None)
        configs.pop(key, None)
    if session is not None:
        previous = session["records"].get((owner, key))
        if previous is not None and previous["status"] in {"tuned", "revalidated"} and status == "hit":
            status = previous["status"]
        session["records"][(owner, key)] = {
            "owner": owner, "key": json.loads(_encode(key)),
            "fingerprint": _fingerprint(owner), "status": status,
            "config": config, "path": str(_profile_path(owner, _encode(key))),
        }
        if config is None and session["mode"] == "require":
            raise RuntimeError(f"missing verified tuning profile: {owner} {_encode(key)}; run prepare_tuning(..., mode='tune') before capture; cache={get_tuning_cache_dir()}")
    elif config is None and owner not in _WARNED:
        warnings.warn(f"FlashRWKV2 {owner} uses an untuned default; run prepare_tuning(..., mode='tune') before capture; cache={get_tuning_cache_dir()}", RuntimeWarning, stacklevel=2)
        _WARNED.add(owner)
    if config is None:
        _WARMED_DEFAULTS.add(identity)
    return default if config is None else config


def save_config(owner, key, config):
    """Atomically publish only after the owner's exact-output validation."""
    encoded = _encode(key)
    path = _profile_path(owner, encoded)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", dir=path.parent, suffix=".tmp", delete=False) as output:
            temporary = output.name
            json.dump({"schema": _SCHEMA, "owner": owner, "fingerprint": _fingerprint(owner),
                       "key": json.loads(encoded), "config": config, "bitwise_equal": True}, output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
    _LOADED.pop((str(_profile_root()), owner, key), None)


def prepare_tuning(calls, *, mode="require", legacy_dir=None):
    """Execute real model calls before capture, loading or tuning their exact keys.

    ``calls`` is an iterable of zero-argument callables. Each callable owns its
    scratch model state and valid metadata; it executes once, so reset state as
    needed before serving. Modes: ``inspect`` reports defaults, ``require`` fails
    at the first missing key, ``tune`` verifies and times only missing keys.
    ``legacy_dir`` is one explicit legacy namespace, revalidated without timing.
    The returned JSON-compatible report describes every actual tuned dispatch.
    """
    if mode not in {"inspect", "require", "tune"}:
        raise ValueError("mode must be inspect, require, or tune")
    if _capturing() or _SESSION.get() is not None:
        raise RuntimeError("prepare_tuning must run outside CUDA Graph capture and other preparation")
    if legacy_dir is not None and mode != "tune":
        raise ValueError("legacy_dir requires mode='tune'")
    if legacy_dir is not None:
        legacy_dir = Path(legacy_dir).expanduser().resolve()
        if not legacy_dir.is_dir():
            raise NotADirectoryError(f"legacy tuning namespace does not exist: {legacy_dir}")
    session = {"mode": mode, "legacy_dir": legacy_dir, "records": {}}
    token = _SESSION.set(session)
    try:
        for call in calls:
            call()
    finally:
        _SESSION.reset(token)
    records = list(session["records"].values())
    if not records:
        raise RuntimeError("prepare_tuning observed no configurable kernel dispatches")
    return {"cache_root": str(get_tuning_cache_dir()), "schema": _SCHEMA,
            "complete": all(row["status"] != "missing" for row in records),
            "counts": {status: sum(row["status"] == status for row in records)
                       for status in ("hit", "missing", "tuned", "revalidated")},
            "records": records}


def tuning_operation(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        if _capturing():
            raise RuntimeError("kernel tuning must finish before CUDA Graph capture")
        token = _TUNING.set(True)
        try:
            return function(*args, **kwargs)
        finally:
            _TUNING.reset(token)
    return wrapped


def revalidation_config():
    return _REVALIDATE.get()


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
    imported = revalidation_config()
    if imported is not None:
        candidates = (imported,)
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
    if imported is not None:
        return eligible[0][0], measurements
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
