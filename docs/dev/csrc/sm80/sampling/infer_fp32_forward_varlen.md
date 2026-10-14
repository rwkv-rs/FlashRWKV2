# Sampling: FP32 packed varlen inference

`infer_fp32_forward_varlen.cpp` keeps the existing registered interfaces for
`setup_sampling_states`, `sampling_temperature_topk_topp_forward_varlen`, and
`sampling_six_parameter_forward_varlen`. Python inputs remain compact FP32
`[B,V]` logits, an explicit Philox slot pool, and unique int32 slot indices.
Scalar and per-request controls share the same filtering algorithm.

The CUDA implementation is adapted from [Rapid-Sampling revision
7da725adbb5afc5b92f6c53383c3999f6809860f](https://github.com/Triang-jyed-driung/Rapid-Sampling/blob/7da725adbb5afc5b92f6c53383c3999f6809860f/sampling.cu).
The imported algorithm uses three histogram passes over the 31 non-sign bits
of nonnegative FP32 probabilities, partitioned as `[10,10,11]`. Padded, sharded
histograms accumulate integer mass and token counts. The cutoff treats equal
probabilities symmetrically by sharing a fractional boundary weight. `top_p=1`
disables the mass cutoff. Filtered probabilities are normalized and converted
to uint32 weights with a total near `4.26e9`; integer prefix scans and rejection
sampling avoid floating CDF rounding and modulo bias. Probabilities below one
quantization unit can become zero. Device arithmetic uses FP32 and integers.

For `B=1,V=65536` without penalties, one cooperative kernel distributes work
across 32 tiles. All tiles use the same softmax maximum and denominator to
preserve ties. Block zero refines the cutoff, using a bounded candidate cache;
cache overflow selects the full vocabulary scan. Cooperative launch is used
only when the device supports it and the exact shared-memory and occupancy
requirements allow all tiles to be resident. Other shapes and penalty calls
use one block per active row. Both paths allocate `[B,V]` FP32 probability and
uint32 weight workspaces. No quantization or histogram tuning parameter is
added to the public API.

FlashRWKV2 retains its request semantics: slot validation runs on device;
invalid metadata and inactive rows return `-1` before reading logits or
advancing RNG. Slot mapping selects both RNG and penalty storage. After a draw,
penalties decay and the selected token receives a frequency increment plus
a presence increment on its first occurrence. Negative zero preserves the
seen marker when increments cancel. This additive rule differs from the
upstream penalty update and is intentionally retained. Scalar temperature
and top-p keep the existing clamping behavior. CUDA Graph replay can change
active count, slots, logits, and per-request controls without recapture.

The RNG storage layout and seeded reproducibility within a launch path remain
supported. Integer sampling changes the token sequence relative to the old
floating CDF implementation. Correctness tests cover fractional distributions,
ties across tiles, cache overflow, slot isolation, penalty updates, and graph
replay. Performance must be measured on the target GPU after correctness;
the upstream speedup is not a measured FlashRWKV2 speedup.

License provenance: the existing FlashRWKV2 adaptation carries MIT notices.
The referenced upstream revision contains no `LICENSE` file or SPDX license
declaration; its license is not supplied by the GitHub repository metadata.
The retained local notice does not establish an upstream license grant.
