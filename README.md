# FlashRWKV2

FlashRWKV2 is a high-performance CUDA operator library for RWKV-7. Its public
inference API exposes the highest semantic fusion islands used by the model;
kernel composition and shape-dependent dispatch remain internal. Complete
models, schedulers, and training frameworks remain the responsibility of
downstream projects.

## Requirements

- Python 3.10 or newer
- Linux, a writable PyTorch extension cache, and a CUDA GPU with Compute
  Capability 8.0 or newer
- A system CUDA Toolkit compatible with PyTorch, a compatible host compiler,
  and Ninja
- `uv`, using the repository-local `./.venv`

The runtime supports PyTorch 2.11 through 2.13. Importing the package does not
compile CUDA code. The first CUDA operator call builds one private ordinary
ATen/pybind extension from the packaged `csrc/sm80` sources for the exact
Compute Capability of all visible GPUs, then reuses PyTorch's extension cache.
All visible GPUs must have the same Compute Capability.

## Installation

The project publishes a universal `py3-none-any` source wheel containing only
Python and CUDA/C++ sources. It contains no precompiled extension or architecture
fallback. Building the wheel does not import Torch or run NVCC:

```bash
python -m pip install --pre FlashRWKV2
```

To prewarm the runtime cache explicitly, use the same build path as the first
operator call:

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

CUDA tests build and load `flashrwkv2._C` lazily and require the toolchain above.

## Benchmarks

Operator benchmarks live in [`benchmarks/`](benchmarks/). For example, run the
WKV7 recurrent correctness benchmark with:

```bash
./.venv/bin/python -m benchmarks.tmix.wkv7.bench \
  --shapes h32d64 \
  --dtype bfloat16 \
  --correctness-only \
  --output /tmp/flashrwkv2-wkv7-correctness.json
```

These benchmarks measure individual operators. They do not report or infer
model-level latency.

The release-specific WKV7 measurements, evidence identities, acceptance
boundaries, and non-target advisory regressions are recorded in the
[FlashRWKV2 0.1.0a9 WKV7 performance report](docs/performance/wkv7-0.1.0a9.md).

For Albatross-compatible TMix low-rank inference, `varlen` means that the
operator consumes packed token rows; it does not mean that one fused kernel is
used for every row count.  The public composite callers automatically use the
canonical fused rank-in window at `M<=7`, the fused rank-out/value-residual
window at `M<=4`, and the canonical large-row linear dispatcher otherwise.
Callers may provide both the original checkpoint layout and a runtime layout.
Both layouts must be prepared outside the timed forward region and retained for
the lifetime of the inference weights; FlashRWKV2 never transposes or copies a
missing layout during dispatch.

TMix inference benchmarks are owned by the public `wkv_prepare` and `readout`
modules. Native-private Linear, activation, VRes, gate and sparse helpers do
not have standalone public benchmark entry points.

## License

FlashRWKV2 is distributed under the
[MIT License](https://github.com/rwkv-rs/FlashRWKV2/blob/main/LICENSE).
