# SPDX-License-Identifier: MIT

"""Benchmark the train_temp L2Wrap CE forward/backward operator family."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import torch
from _timing import measure_cuda

from flashrwkv2.loss.l2wrap_ce import pretrain_l2wrap_ce_bf16
from flashrwkv2.loss.l2wrap_ce.tilelang import L2Wrap

SOURCE_REVISION = "952102498e9ed367ea0a59ee64106916d474d30f"
SOURCE_PATHS = (
    "flashrwkv2/loss/l2wrap_ce/tilelang.py",
    "flashrwkv2/_internal/logprob.py",
)


def _source_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for relative in SOURCE_PATHS:
        path = root / relative
        digest.update(relative.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _git_metadata(root: Path) -> dict[str, str]:
    def run(*args: str) -> str:
        result = subprocess.run(
            args, cwd=root, check=False, capture_output=True, text=True
        )
        return result.stdout.strip()

    return {
        "revision": run("git", "rev-parse", "HEAD"),
        "status": run("git", "status", "--short"),
    }


def _metadata() -> dict[str, object]:
    capability = torch.cuda.get_device_capability()
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": f"{capability[0]}.{capability[1]}",
        "total_memory_bytes": properties.total_memory,
        "torch": torch.__version__,
        "cuda_runtime": str(torch.version.cuda),
        "python": platform.python_version(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--vocab", type=int, default=128)
    parser.add_argument("--output", type=str, default="/tmp/flashrwkv2-l2wrap-ce.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.rows <= 0 or args.vocab <= 0:
        raise ValueError("rows and vocab must be positive")

    device = torch.device("cuda")
    torch.manual_seed(17)
    logits = (
        torch.randn(args.rows, args.vocab, device=device, dtype=torch.float32)
        .mul_(0.2)
        .requires_grad_()
    )
    targets = torch.randint(args.vocab, (args.rows,), device=device, dtype=torch.int64)

    pretrain_l2wrap_ce_bf16(logits, targets)
    grad_loss = torch.ones((), device=device, dtype=torch.float32)

    def launch() -> None:
        loss = L2Wrap.apply(logits, targets)
        torch.autograd.grad(loss, logits, grad_loss)

    timing = measure_cuda(launch)

    root = Path(__file__).resolve().parents[3]
    result = {
        "operator": "pretrain_l2wrap_ce_bf16",
        "rows": args.rows,
        "vocab": args.vocab,
        "profile": f"rows={args.rows}/vocab={args.vocab}",
        "timing_boundary": "tilelang_forward_backward",
        "steady_state": "FP32 forward intermediates are recreated for each launch",
        **timing,
        "source_repository": "https://github.com/BlinkDL/RWKV-LM",
        "source_revision": SOURCE_REVISION,
        "source_paths": SOURCE_PATHS,
        "source_hash": _source_hash(root),
        "compiled_extension_hash": None,
        **_metadata(),
        "git": _git_metadata(root),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
