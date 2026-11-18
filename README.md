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
capturing. Dense and packed training accept optional FP32 initial state and can
return final state for the next chunk. See the [training contract](docs/kernel_api.md#training).

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
Trusted repository PRs run on the self-hosted GPU. Daily runs cover the main branch.

For a release, dispatch **Quality Gate** on the release tag to include Compute Sanitizer
memcheck and racecheck. Download its `dist-<commit>` artifact to `artifacts/dist` on the
RTX 4090 checkout of that tag and use the same wheel for independent acceptance:

```bash
timeout 600s uv sync --frozen --no-install-project --group dev
timeout 120s uv pip install --python .venv/bin/python --no-deps --reinstall artifacts/dist/*.whl
PATH="$PWD/.venv/bin:$PATH" timeout 18500s .venv/bin/python -I ci/quality_gate.py validate \
  --release --dist artifacts/dist --output artifacts/sm89
```

Attach `artifacts/sm89/sm89.json` to the draft release. The manual **PyPI release**
workflow verifies the tag, main ancestry, both GPU reports and the SHA-256 of each
original distribution. Publishing the GitHub release publishes those files using PyPI
trusted publishing and checks their public digests.

## Benchmarks

Operator benchmarks live in [`benchmarks/`](benchmarks/). The WKV matrix covers
batch size, sequence length, and head dimensions:

```bash
PYTHONPATH=benchmarks ./.venv/bin/python benchmarks/tmix/wkv7/bench.py \
  --shapes h12d64 h16d64 h32d64 h40d64 h64d64 h80d64 h16d128 h8d256 \
  --cases 1x1 16x1 1x16 64x16 1x128 --tune --timing-mode graph \
  --output artifacts/wkv7-recurrent-bf16.json
```

RTX 4090 and PRO 6000 use checked-in execution configurations. Each operator benchmark
can evaluate alternatives with `--tune`. It verifies outputs and state against the
canonical implementation before measuring CUDA Graph replay. Warm each graph shape with
its real dtype, layouts and gradient requirements before capture and restore caller-owned
state after warmup. See [execution configuration](docs/kernel_api.md#execution-configuration).

## License

FlashRWKV2 is distributed under the
[MIT License](https://github.com/rwkv-rs/FlashRWKV2/blob/main/LICENSE).
