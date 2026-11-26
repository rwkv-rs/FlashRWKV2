# CI Quality Gate & Release Acceptance

This document describes the maintainer workflow for CI validation, dual-GPU release acceptance (SM120 PRO 6000 + SM89 RTX 4090), and PyPI publication.

---

## Quality Gate Overview

The required **Quality Gate** workflow (`.github/workflows/pro6000-gpu.yml`) runs on pull requests, pushes to `main`, daily schedules, and release dispatches:

1. **`package` job (Ubuntu CPU)**:
   - Builds the universal `py3-none-any` wheel and sdist via `uv build`.
   - Runs `ruff check ci benchmarks` and `ruff check --select E9,F63,F7,F82 flashrwkv2 tests`.
   - Runs CPU contract tests (`pytest -q -m 'not cuda and not resource'`).
   - Uploads `dist-<commit>` artifacts.
2. **`gpu` job (Self-hosted PRO 6000, SM120)**:
   - Installs the exact built wheel into `./.venv`.
   - Runs `ci/quality_gate.py validate` covering the full CUDA pytest suite, resource checks, and every operator benchmark.
   - Uploads JUnit XML, device metadata (`sm120.json`), and benchmark logs.
3. **Docs-only fast path**:
   - Pull requests modifying only `README.md` or `docs/**/*.md` pass with the `package` job without blocking on the self-hosted GPU runner.
   - Runtime changes require a trusted repository branch for the self-hosted GPU check.

---

## Release Acceptance Procedure

Releases require independent verification of the **exact same wheel** on both **SM120 (PRO 6000, 188 SMs)** and **SM89 (RTX 4090, 128 SMs)**, including the PyTorch 2.11–2.14 compatibility matrix and Compute Sanitizer (`memcheck` and `racecheck`).

### 1. Run Tagged Quality Gate on SM120

Dispatch **Quality Gate** (`pro6000-gpu.yml`) on the release tag `v<version>` (must match `pyproject.toml` and `flashrwkv2.__version__` and be an ancestor of `origin/main`). This runs `--release` mode on PRO 6000 and uploads `dist-<commit>` and `gpu-<commit>` (containing `sm120.json`).

### 2. Run Independent Acceptance on SM89 (RTX 4090)

Check out the release tag on the RTX 4090 machine, download the `dist-<commit>` artifact into `artifacts/dist`, and execute:

```bash
timeout 600s uv sync --frozen --no-install-project --group dev
timeout 120s uv pip install --python .venv/bin/python --no-deps --reinstall artifacts/dist/*.whl
PATH="$PWD/.venv/bin:$PATH" timeout 18500s .venv/bin/python -I ci/quality_gate.py validate \
  --release --dist artifacts/dist --output artifacts/sm89
```

### 3. Verify Both Reports

Download `sm120.json` from the tagged `gpu-<commit>` artifact into `artifacts/sm89` and verify both reports against the distribution hashes:

```bash
timeout 60s .venv/bin/python -I ci/quality_gate.py verify \
  --dist artifacts/dist --output artifacts/sm89
```

### 4. Publish Release

1. Attach `artifacts/sm89/sm89.json` to the draft GitHub release for `v<version>`.
2. Publish the release.
3. The **PyPI release** workflow (`.github/workflows/release.yml`) automatically verifies the tag identity, `main` ancestry, both GPU reports, and original SHA-256 digests, publishes via PyPI Trusted Publishing, and checks the public PyPI hashes.

---

## Historical Acceptance Records

### Validation Summary (SM89 & SM120)

Business tests cover train/prefill/decode equivalence, independent VJPs and optimizer updates, state ownership, frozen/unused gradients, `torch.compile`/CUDA Graph execution, dynamic workloads, and fresh-process disk-cache reuse. Reference RWKV-7 geometries cover `C = 2048 / 2560 / 4096 / 4096` with `W/A/G/V` low-rank widths `96/96/256/64`, `96/96/320/64`, `128/128/480/96`, and `192/192/384/128`.

| Measured Result | SM89 (RTX 4090) | SM120 (PRO 6000) |
| --- | --- | --- |
| Business GPU suite (complete block, numerics, state, Graph, prepared workloads) | 330 passed | 330 passed |
| Training workloads within 2% of best measured comparison | 9 / 9 | 9 / 9 |
| Fresh-process disk-cache reuse | 139 cold / 0 warm | 137 cold / 0 warm |

### SM120 Training Acceptance Detail

Measured on an idle Shanghai PRO 6000 Workstation (188 SMs), PyTorch 2.13.0+cu130, CUDA 13.0, using paired ABBA timing (7 samples × 5 Graph replays per phase, excluding preparation and capture; B=1 layer + LM Head, forward + backward + grad zeroing, excluding optimizer):

| Workload | Best Comparison (ms) | Prepared (ms) | Latency Change | Retained Gain |
| --- | ---: | ---: | ---: | ---: |
| 7.2B T1024 eager | 32.786 | 32.196 | -1.80% | 119.77% |
| 13.3B T128 eager | 7.399 | 7.311 | -1.20% | 114.17% |
| 13.3B T1024 eager | 32.897 | 32.267 | -1.91% | 121.38% |
| 1.5B T128 compiled | 3.603 | 3.577 | -0.71% | 109.19% |

SM89 T1024 training recorded 42.20 ms (7.2B) and 42.21 ms (13.3B) (−0.39% / +0.33% vs baseline). Gain retention applies when `default - best > max(0.02 * default, 3 * (IQR(default) + IQR(best)))`. Performance comparisons always require idle hardware, bitwise numerical checks, and paired timing; unit-test wall time is not performance evidence.
