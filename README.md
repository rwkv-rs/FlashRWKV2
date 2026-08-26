# FlashRWKV2

FlashRWKV2 is a high-performance CUDA operator library for RWKV-7. Its public
inference API exposes the highest semantic fusion islands used by the model;
kernel composition and shape-dependent dispatch remain internal. Complete
models, schedulers, and training frameworks remain the responsibility of
downstream projects.

## Requirements

- Python 3.10 or newer
- An NVIDIA CUDA build environment and a supported CUDA device
- `uv`, using the repository-local `./.venv`
- A CUDA GPU binary-compatible with the native SM90 or SM120 backends

The runtime and reproducible source-build contracts currently pin PyTorch
2.13.0. Native builds always compile separate SM90 and SM120 private extensions;
the active extension is selected at runtime. A backend suffix is its minimum
native cubin target, not an exact-device allowlist: `_C_sm90` serves compatible
SM9.x devices and `_C_sm120` serves compatible SM12.x devices whose minor
compute capability is at least the compiled target.

## Installation

The current alpha publishes a prebuilt CPython 3.12 Linux x86_64 wheel. Its
currently validated product is the RTX PRO 6000 at SM120; binary compatibility
does not by itself constitute a product-level correctness or CUDA Graph claim.
The wheel requires glibc 2.38 or newer, PyTorch 2.13.0, and a CUDA 13 runtime
supplied through PyTorch's dependencies:

```bash
python -m pip install --pre FlashRWKV2
```

Other Python or platform combinations install from the source distribution and
build both CUDA extensions on the target machine:

```bash
python -m pip install --pre FlashRWKV2
```

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

CUDA tests require a successfully built `flashrwkv2._C` extension and a
supported GPU.

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
