# FlashRWKV2

FlashRWKV2 is a TileLang operator library for RWKV-7. Training and inference use
BF16 token inputs and outputs with FP32 accumulation. Training uses FP32 recurrent state. Its public
inference API exposes the highest semantic fusion islands used by the model;
kernel composition and shape-dependent dispatch remain internal. Complete
models, schedulers, and training frameworks remain the responsibility of
downstream projects.

For serving on SM89 and SM120, prefer FP16 recurrent state for both prefill and
decode. This path uses BF16 token IO and FP32 arithmetic, and rounds state to
FP16 after every token. For D64 with 40 or 64 heads, prefill and decode use the same
four groups of 16 keys for each dot product. A live metadata ticket schedules longer
prefill rows first, and each decode slot can append to its DeltaLog independently.
With one to three prefill rows, four blocks share each head's value columns.
With four or more prefill rows, one block handles a whole head.
On the 128-SM RTX 4090, H64 stores up to two exact DeltaLog records per head
and slot (1,540 bytes). The other H40/H64 configurations store one record (1,284 bytes). Use the handle's `memory_layout`
when budgeting concurrency. Call `materialize_slots_(indices)` before reading
or copying the externally bound tensor. Training, RL rollout and its evaluation
and sampling search use FP32 state to reproduce the learner's logprobs.

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

Warm the forward and backward on the capture stream with the actual shapes,
then restore state and gradients before capture. Dense and packed training
accept optional FP32 initial state and can return final state for the next
chunk. See the [training contract](docs/kernel_api.md#training).

Speculative decoding can export per-token WKV checkpoints and normalized TMix
and CMix shift states from the verifier. The optional
`flashrwkv2.mtp.DraftLoop` runs a single-request LightMTP block eagerly or in
a CUDA Graph. See the [state export and DraftLoop contracts](docs/kernel_api.md#state-and-packed-metadata)
and the [original TileRWKV changes by Joluck](https://github.com/Joluck/tilerwkv/commit/409b492e952086a61d89817d8032375a851b51da).

## Tests

```bash
./.venv/bin/python -m pytest -q
```

CUDA tests compile TileLang kernels and the metadata/sampling extension lazily.
The full block suite checks outputs, FP32 state, and selected logprobs bitwise
across training/inference, prefill/decode, and batch composition.

## CI and release acceptance

The required **Quality Gate** builds the wheel and sdist, runs CPU contracts, and runs
pytest and every operator benchmark on the PRO 6000 against the installed wheel.
Each run uploads its distributions, JUnit XML, device metadata and benchmark results.
Documentation-only PRs can pass with the package and CPU checks. Runtime changes
require a trusted repository branch for the self-hosted GPU check. Daily runs cover
the main branch.

For a release, dispatch **Quality Gate** on the release tag for the Torch 2.11–2.14 compatibility matrix and
Compute Sanitizer memcheck and racecheck. Download its `dist-<commit>` artifact to `artifacts/dist` on the
RTX 4090 checkout of that tag and use the same wheel for independent acceptance:

```bash
timeout 600s uv sync --frozen --no-install-project --group dev
timeout 120s uv pip install --python .venv/bin/python --no-deps --reinstall artifacts/dist/*.whl
PATH="$PWD/.venv/bin:$PATH" timeout 18500s .venv/bin/python -I ci/quality_gate.py validate \
  --release --dist artifacts/dist --output artifacts/sm89
```

Download `sm120.json` from the tagged GPU artifact into `artifacts/sm89` and
verify both reports before publishing the draft release:

```bash
timeout 60s .venv/bin/python -I ci/quality_gate.py verify \
  --dist artifacts/dist --output artifacts/sm89
```

Attach `artifacts/sm89/sm89.json` to the draft release. On publication, the
**PyPI release** workflow verifies the tag, main ancestry, both GPU reports and
the original distribution hashes. It publishes through PyPI trusted publishing
and checks the public digests.

## Benchmarks

Run the LightMTP comparison at the RWKV-7 G1k 7.2B draft shape with:

```bash
PYTHONPATH=benchmarks timeout 10m .venv/bin/python benchmarks/mtp/bench.py --depth 4
```

Operator benchmarks live in [`benchmarks/`](benchmarks/). The WKV matrix covers
batch size, sequence length, and head dimensions:

```bash
PYTHONPATH=benchmarks ./.venv/bin/python benchmarks/tmix/wkv7/bench.py \
  --shapes h12d64 h16d64 h32d64 h40d64 h64d64 h80d64 h16d128 h8d256 \
  --cases 1x1 16x1 1x16 64x16 1x128 --tune --timing-mode graph \
  --output artifacts/wkv7-recurrent-bf16.json
```

To measure chunked serving with decode rows first and randomly assigned slots:

```bash
PYTHONPATH=benchmarks ./.venv/bin/python benchmarks/tmix/wkv7/bench.py \
  --shapes h64d64 --state-dtype float16 --timing-mode graph --metadata live \
  --slot-order random --state-pool-size 320 --cases 255x1+1x4 255x1+1x256 1x1024 \
  --output artifacts/wkv7-live.json
```

The per-layer time uses the ticket shared by the model's layers. Add
`--include-ticket-preparation` to capture and time the device-side validation and sort
together with one WKV layer on every step.

RTX 4090 and PRO 6000 use checked-in execution configurations. Each operator benchmark
can evaluate alternatives with `--tune`. It verifies outputs and state against the
canonical implementation before measuring CUDA Graph replay. Warm each graph shape with
its real dtype, layouts and gradient requirements before capture and restore caller-owned
state after warmup. See [execution configuration](docs/kernel_api.md#execution-configuration).

## License

FlashRWKV2 is distributed under the
[MIT License](https://github.com/rwkv-rs/FlashRWKV2/blob/main/LICENSE).
