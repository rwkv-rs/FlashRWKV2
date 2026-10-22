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
- Linux, a writable PyTorch extension cache, and a CUDA GPU with Compute
  Capability 8.0 or newer
- A system CUDA Toolkit compatible with PyTorch, a compatible host compiler,
  and Ninja
- `uv`, using the repository-local `./.venv`

The runtime supports PyTorch 2.11 through 2.13 and pins TileLang 0.1.15.
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

## Tests

```bash
./.venv/bin/python -m pytest -q
```

CUDA tests compile TileLang kernels and the metadata/sampling extension lazily.
The full block suite checks outputs, FP32 state, and selected logprobs bitwise
across training/inference, prefill/decode, and batch composition.

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
The historical [0.1.0a9 performance report](docs/performance/wkv7-0.1.0a9.md)
describes that release's implementation; current results must identify the
revision, hardware, shapes, and numerical contract.

## License

FlashRWKV2 is distributed under the
[MIT License](https://github.com/rwkv-rs/FlashRWKV2/blob/main/LICENSE).
