# Release Acceptance

Maintainers run validation and publish to PyPI from the release checkout.

## Source Checks

```bash
timeout 600s uv sync --frozen --group dev
timeout 60s .venv/bin/ruff check ci benchmarks
timeout 60s .venv/bin/ruff check --select E9,F63,F7,F82 flashrwkv2 tests
CUDA_VISIBLE_DEVICES="" timeout 600s .venv/bin/python -m pytest -q -m 'not cuda and not resource'
```

## Release Acceptance Procedure

Releases require independent verification of the **exact same wheel** on both **SM120 (PRO 6000, 188 SMs)** and **SM89 (RTX 4090, 128 SMs)**, including the PyTorch 2.11–2.14 compatibility matrix and Compute Sanitizer (`memcheck` and `racecheck`).

### 1. Build the Release Distributions

Check out the release tag. Its version must match `pyproject.toml` and `flashrwkv2.__version__`, and its commit must be an ancestor of `origin/main`. Build the wheel and sdist once into an empty `artifacts/release/dist` directory.

```bash
timeout 60s git fetch origin main
timeout 30s git merge-base --is-ancestor HEAD origin/main
timeout 300s uv build --out-dir artifacts/release/dist
```

### 2. Validate on SM120 and SM89

Check out the same tag and copy the original distribution files into `artifacts/release/dist` on each GPU host. With the target GPU visible and CUDA's `compute-sanitizer` on `PATH`, run the acceptance script on each host.

```bash
timeout 600s uv sync --frozen --no-install-project --group dev
timeout 120s uv pip install --python .venv/bin/python --no-deps --reinstall artifacts/release/dist/*.whl
PATH="$PWD/.venv/bin:$PATH" timeout 18500s .venv/bin/python -I ci/quality_gate.py validate \
  --release --dist artifacts/release/dist --output artifacts/release/results
```

### 3. Verify Both Reports

Collect `sm120.json` and `sm89.json` in `artifacts/release/results` on the release checkout. Verify that both reports match the tagged commit and distribution hashes.

```bash
timeout 60s .venv/bin/python -I ci/quality_gate.py verify \
  --dist artifacts/release/dist --output artifacts/release/results
```

### 4. Publish Release

Provide a project-scoped PyPI API token through `UV_PUBLISH_TOKEN` and upload the verified distributions.

```bash
timeout 180s uv publish artifacts/release/dist/*.whl artifacts/release/dist/*.tar.gz
```

Compare the public PyPI SHA-256 digests with the verified reports. Attach both reports to the GitHub release for the tag.

---

## Historical Acceptance Records

> The measurements below are historical acceptance records from commits `a239bc3` and `07c7913`. Raw paired samples and numerical evidence live in `artifacts/cmix-acceptance-a239bc3.json` and `artifacts/sm120-remaining-acceptance-results-07c7913.json`. Validate new runtime changes on the final candidate commit before merge or release.

### Validation Summary (SM89 & SM120, historical commits `a239bc3` / `07c7913`)

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
