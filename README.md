# FlashRWKV2

FlashRWKV2 is a TileLang operator library for RWKV-7 with a shared BF16
training and inference forward, FP32 accumulation, and FP32 recurrent state. Its public
inference API exposes the highest semantic fusion islands used by the model;
kernel composition and shape-dependent dispatch remain internal. Complete
models, schedulers, and training frameworks remain the responsibility of
downstream projects.

An optional FP16-state inference mode halves state storage while keeping BF16
token IO and FP32 arithmetic. It does not promise bitwise equality with FP32
training. The default FP32-state path retains training precision and parity.

## Requirements

- Python 3.10 or newer
- Linux, a writable FlashRWKV2 cache, and a CUDA GPU with Compute
  Capability 8.0 or newer
- A system CUDA Toolkit compatible with PyTorch, a C++20 host compiler,
  and Ninja
- `uv`, using the repository-local `./.venv`

The runtime supports PyTorch 2.11 through 2.14 and pins TileLang 0.1.15.
Importing the package does not compile kernels. TileLang specializes model
operators for the active GPU and caches compiled kernels. Packed metadata
validation and sampling use a small ATen/pybind extension, compiled from the
packaged native sources for the Compute Capability of the visible GPUs.
All GPUs visible to that extension must have the same Compute Capability.

## Installation

The project publishes a universal `py3-none-any` source wheel containing only
Python and CUDA/C++ sources. It contains no precompiled extension or architecture
fallback. Building the wheel does not import Torch or run NVCC:

```bash
python -m pip install --pre FlashRWKV2
```

To prewarm the metadata and sampling extension cache:

```bash
python -m flashrwkv2.compile
```

The command emits JSON containing `status` (`compiled` or `cached`), `target`,
`cache_key`, and `library`. Build failures preserve the original compiler
diagnostics; FlashRWKV2 does not download a compiler or select another backend.

For development from a checkout:

```bash
git clone https://github.com/rwkv-rs/FlashRWKV2.git
cd FlashRWKV2
uv sync
./.venv/bin/python -m pip install -v --no-build-isolation -e .
```

## Kernel API

See the [Kernel API reference](https://github.com/rwkv-rs/FlashRWKV2/blob/main/docs/kernel_api.md)
for the complete public operator surface and tensor contracts.

Training operators support `torch.compile(fullgraph=True)`, including backward,
LoRA/NoRA, and CUDA Graph capture. Preserve the BF16 intermediate rounding
contract with Inductor's precision option:

```python
compiled_model = torch.compile(
    model, fullgraph=True, options={"emulate_precision_casts": True},
)
```

Prepare tuning with eager forward-and-backward calls before compiling or
capturing. StateTune chunk metadata is prepared once with
`prepare_tmix_wkv7_statetune_metadata` and supplied through `prepared_metadata`;
see the [training contract](docs/kernel_api.md#training).

## Tests

```bash
./.venv/bin/python -m pytest -q
```

CUDA tests compile TileLang kernels and the metadata/sampling extension lazily.
The full block suite checks outputs, FP32 state, and selected logprobs bitwise
across training/inference, prefill/decode, and batch composition.

## CI and release acceptance

`Quality Gate` is the required merge check. Ordinary PRs run CPU contracts,
package checks when needed, and the affected owners' complete CUDA test matrices
on PRO6000 with Torch 2.14.1. Provider changes expand to their consumers and the
block integration suite. Kernel changes also run the selected memcheck/racecheck
cases. Changes to dependencies, native interfaces, or runtime loading add the
Torch 2.11–2.14 compatibility matrix. Tests load the installed source wheel;
pytest's checkout import path is excluded from GPU validation.

Daily validation runs at 08:17 UTC on `main`, with full correctness, CUDA Graph,
memcheck, and racecheck coverage. Sunday adds all supported Torch versions, a
cold native build, and the complete benchmark/tuning suite. Daily performance
reports measure owners changed since their latest compatible successful
measurement. Performance failures and regressions appear in an independent
report; they do not block the PR quality check. The Actions summary records
actual selected/executed tests and tracks five completed PR timings against an
initial 20-minute target for a single-owner change, excluding queue time.

Successful results and original distributions share one 90-day artifact bundle.
Identical source trees can reuse CPU results and individual GPU version results
when the policy, environment, and complete collected test scope match. A full run can add
missing compatibility or cold-build checks to an existing bundle. Validation
never combines module coverage from different source revisions.

Before releasing, run the `Quality Gate` workflow on the exact release tag with
`mode=full` and `extended=true`. This produces the original wheel/sdist, full
PRO6000 correctness and compatibility results, and a separate complete
performance report. The `PyPI release` workflow verifies these results and
publishes those original files. Its manual `tag` input verifies acceptance
without publishing.

RTX4090 acceptance is independent. On its checkout of the same tag, download
the PRO6000 `flashrwkv2-quality-v7-<source-tree>` artifact to
`artifacts/acceptance`. Use the project `.venv`, with the default Torch version,
and the wheel from that bundle. With read access to Actions available through
`GH_TOKEN`, the existing CI entrypoints produce the independent reports:

```bash
timeout 600s uv sync --frozen --no-install-project --group dev
timeout 60s .venv/bin/python ci/quality_gate.py plan \
  --base HEAD^ --head HEAD --mode full --extended --target sm89 \
  --output artifacts/sm89/plan.json
timeout 60s .venv/bin/python ci/quality_gate.py package \
  --directory artifacts/acceptance --output artifacts/sm89/package.json
CUDA_VISIBLE_DEVICES="" timeout 600s .venv/bin/python ci/quality_gate.py test \
  --plan artifacts/sm89/plan.json --suite cpu --output artifacts/sm89/cpu.json \
  --reuse artifacts/acceptance/quality-manifest.json
CUDA_VISIBLE_DEVICES=0,1 PATH="$PWD/.venv/bin:$PATH" timeout 6600s .venv/bin/python ci/quality_gate.py gpu \
  --plan artifacts/sm89/plan.json --package artifacts/acceptance \
  --python .venv/bin/python --output artifacts/sm89
# Compatibility testing ends on an older Torch; restore the primary version.
timeout 600s uv sync --frozen --no-install-project --group dev
timeout 180s uv pip install --python .venv/bin/python --no-deps --reinstall artifacts/acceptance/dist/*.whl
timeout 120s .venv/bin/python ci/quality_gate.py finalize \
  --plan artifacts/sm89/plan.json --package artifacts/acceptance \
  --validation artifacts/sm89 --output artifacts/sm89-quality-manifest.json
CUDA_VISIBLE_DEVICES=0,1 PATH="$PWD/.venv/bin:$PATH" timeout 13800s .venv/bin/python ci/benchmark_gate.py report \
  --quality artifacts/sm89-quality-manifest.json --package artifacts/acceptance \
  --profile full --repository rwkv-rs/FlashRWKV2 --python .venv/bin/python \
  --output artifacts/sm89/performance --summary artifacts/sm89/performance.md
timeout 30s cp artifacts/sm89/performance/summary.json artifacts/sm89-performance-summary.json
```

Attach `sm89-quality-manifest.json` and `sm89-performance-summary.json` to the
tag's draft GitHub release. Both identify the exact source tree and wheel
SHA256; the release verifier checks RTX4090 identity, full test completion,
compatibility, sanitizer coverage, and complete performance results on both
architectures. Publish the release after its manual verification succeeds.

## Benchmarks

Operator benchmarks live in [`benchmarks/`](benchmarks/). The WKV matrix covers
batch size, sequence length, and head dimensions:

```bash
PYTHONPATH=benchmarks ./.venv/bin/python benchmarks/tmix/wkv7/bench.py \
  --shapes h12d64 h16d64 h32d64 h40d64 h64d64 h80d64 h16d128 h8d256 \
  --cases 1x1 16x1 1x16 64x16 1x128 --tune --timing-mode graph \
  --output artifacts/wkv7-recurrent-bf16.json
```

Tuning candidates are admitted only after exact-output checks. Performance
acceptance requires correctness evidence and measurements on both RTX 4090
(SM89) and PRO6000 (SM120). Operator timing does not establish model latency.
PostNorm, TMix inference stages, CMix and Head linear benchmarks support
`--implementation cuda-baseline`
when `PYTHONPATH` selects the original CUDA package. Run the same benchmark
script and seed for both revisions; emitted JSON records their actual precision.
`--timing-mode eager` measures dispatch wall time and `graph` measures a captured
batch with CUDA events. Compare the two timing modes separately. PostNorm,
TokenShift and Readout benchmarks also expose `--tune`.
Verified profiles and native builds use `~/.cache/flashrwkv2`, independent of
`TORCH_HOME`, `XDG_CACHE_HOME`, virtual environments and release directories.
Set `FLASHRWKV2_CACHE_DIR` before importing FlashRWKV2 to explicitly relocate it.
TileLang manages its compiled-kernel cache separately. Native extension cache
hits describe metadata and sampling, while model tuning coverage is checked by
running representative calls through the public preparation API:

```python
from flashrwkv2 import prepare_tuning

# Each callable uses valid metadata and disposable/reset model state.
# Include every prefill/decode shape, graph batch and weight layout in service.
# Training callables also run backward with the actual trainable parameters.
report = prepare_tuning(warmup_calls, mode="inspect")
report = prepare_tuning(warmup_calls, mode="tune")  # only missing keys
report = prepare_tuning(warmup_calls, mode="require")  # before graph capture
```

`inspect` reports exact dispatch coverage; `require` loads verified selections
and fails on missing keys; `tune` verifies output/state before timing candidates.
The report includes each owner, device class, full key, source fingerprint,
selected configuration and profile path. Same-model GPUs with the same SM count
reuse selections across GPU ordinals. Concurrent preparation serializes each
missing key and rechecks after taking its lock. Ordinary forwards warn about
untuned defaults and never start tuning. A previously missing file is rechecked
on later eager calls; existing CUDA Graphs must be recaptured after preparation.
Preparation is an exclusive phase within each process and includes dispatches
from autograd worker threads. Finish serving/training calls before preparing.
To migrate a legacy namespace, pass its full directory as `legacy_dir` with
`mode="tune"`: matching configurations are revalidated on current inputs without
benchmarking, while missing shapes are tuned normally. See the
[tuning contract](docs/kernel_api.md#tuning-and-acceptance) for compatibility,
state ownership and deployment examples.
The historical [0.1.0a9 performance report](docs/performance/wkv7-0.1.0a9.md)
describes that release's implementation; current results must identify the
revision, hardware, shapes, and numerical contract.

## License

FlashRWKV2 is distributed under the
[MIT License](https://github.com/rwkv-rs/FlashRWKV2/blob/main/LICENSE).
