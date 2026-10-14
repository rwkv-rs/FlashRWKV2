// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright contributors to Rapid-Sampling
// Adapted from Rapid-Sampling revision 7da725adbb5afc5b92f6c53383c3999f6809860f.
// Slot lifecycle and additive penalty semantics are owned by FlashRWKV2.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace {

using RAND = curandStatePhilox4_32_10_t;
constexpr int RAND_SETUP_THREADS = 256;
struct HistogramLayout {
  int passes, max_bins;
  int widths[3], shifts[3];
  int shards = 2;
};

__device__ __forceinline__ float safe_float(float x) {
  return isnan(x) ? 0.f : (isinf(x) ? copysignf(FLT_MAX, x) : x);
}

template <typename T, bool Maximum>
__device__ __forceinline__ T warp_reduce(T value) {
  if constexpr (Maximum && std::is_same<T, float>::value) {
    unsigned bits = __float_as_uint(value);
    unsigned ordered = (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
    ordered = __reduce_max_sync(0xffffffff, ordered);
    return __uint_as_float((ordered & 0x80000000u) ? (ordered ^ 0x80000000u) : ~ordered);
  } else if constexpr (!Maximum && std::is_same<T, unsigned>::value) {
    return __reduce_add_sync(0xffffffff, value);
  } else {
#pragma unroll
    for (int offset = 16; offset; offset >>= 1) {
      T other = __shfl_down_sync(0xffffffff, value, offset);
      if constexpr (Maximum)
        value = value > other ? value : other;
      else
        value += other;
    }
    return value;
  }
}

template <typename T, bool Maximum = false>
__device__ T block_reduce(T value, T* scratch) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  value = warp_reduce<T, Maximum>(value);
  if (lane == 0) scratch[warp] = value;
  __syncthreads();
  if (warp == 0) {
    T total = lane < blockDim.x / 32 ? scratch[lane] : T(0);
    if constexpr (Maximum) {
      if (lane >= blockDim.x / 32) total = T(-FLT_MAX);
    }
    total = warp_reduce<T, Maximum>(total);
    if (lane == 0) scratch[0] = total;
  }
  __syncthreads();
  T total = scratch[0];
  __syncthreads();  // Prevent a subsequent reduction from overwriting readers' data.
  return total;
}

template <typename T>
__device__ T block_scan(T value, T* scratch) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    T other = __shfl_up_sync(0xffffffff, value, offset);
    if (lane >= offset) value += other;
  }
  if (lane == 31) scratch[warp] = value;
  __syncthreads();
  if (warp == 0) {
    T total = lane < blockDim.x / 32 ? scratch[lane] : T(0);
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
      T other = __shfl_up_sync(0xffffffff, total, offset);
      if (lane >= offset) total += other;
    }
    if (lane < blockDim.x / 32) scratch[lane] = total;
  }
  __syncthreads();
  if (warp) value += scratch[warp - 1];
  __syncthreads();
  return value;
}

__global__ void create_rng_kernel(RAND* states, unsigned long long seed, int64_t count) {
  int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) curand_init(seed, i, 0, states + i);
}

__device__ unsigned random_below32(RAND* state, unsigned total) {
  unsigned value = curand(state);
  unsigned high = __umulhi(value, total), low = value * total;
  if (low < total) {
    const unsigned cutoff = (0u - total) % total;
    while (low < cutoff) {
      value = curand(state);
      high = __umulhi(value, total);
      low = value * total;
    }
  }
  return high;
}

struct SamplingControls {
  float presence = 0.f, frequency = 0.f, decay = 1.f;
  float temperature = 1.f, top_p = 1.f;
  int top_k = -1;
  const float *presences = nullptr, *frequencies = nullptr, *decays = nullptr;
  const float *temperatures = nullptr, *top_ps = nullptr;
  const int* top_ks = nullptr;
};

__device__ __forceinline__ SamplingControls sampling_row_controls(SamplingControls controls,
                                                                  int row, int vocab_size) {
  controls.presence = controls.presences ? controls.presences[row] : controls.presence;
  controls.frequency = controls.frequencies ? controls.frequencies[row] : controls.frequency;
  controls.decay = controls.decays ? controls.decays[row] : controls.decay;
  controls.temperature =
      fminf(fmaxf(controls.temperatures ? controls.temperatures[row] : controls.temperature, .001f),
            1000.f);
  controls.top_k = controls.top_ks ? controls.top_ks[row] : controls.top_k;
  controls.top_p = fminf(fmaxf(controls.top_ps ? controls.top_ps[row] : controls.top_p, 0.f), 1.f);
  if (controls.top_k <= 0 || controls.top_k > vocab_size) controls.top_k = vocab_size;
  if (controls.top_p == 0.f) {
    controls.top_k = 1;
    controls.top_p = 1.f;
  }
  return controls;
}

template <bool Penalties, bool Cooperative = false, int CachedCandidates = 0>
__device__ __forceinline__ void sampling_body(int V, const float* logits, float* penalties,
                                              int* outputs, RAND* states, float* probs,
                                              unsigned* weights, float inv_temperature, int top_k,
                                              float top_p, float presence, float repetition,
                                              float decay, HistogramLayout layout,
                                              const unsigned* first_histogram = nullptr,
                                              const int* slot_indices = nullptr) {
  const int t = threadIdx.x, b = blockIdx.x;
  const int slot = slot_indices == nullptr ? b : slot_indices[b];
  states += slot;
  logits += int64_t(b) * V;
  probs += int64_t(b) * V;
  weights += int64_t(b) * V;
  if constexpr (Penalties) penalties += int64_t(slot) * V;
  __shared__ float fscratch[1024 / 32];
  __shared__ unsigned iscratch[1024 / 32];
  const int histogram_stride = layout.max_bins + 1;
  constexpr float mass_scale = 0x1p31f;
  extern __shared__ __align__(8) unsigned char histogram_storage[];
  unsigned* mass = reinterpret_cast<unsigned*>(histogram_storage);
  unsigned* counts = reinterpret_cast<unsigned*>(mass + layout.shards * histogram_stride);
  float* candidate_cache = reinterpret_cast<float*>(counts + layout.shards * histogram_stride);
  unsigned candidate_count = 0, candidate_overflow = 0;
  __shared__ unsigned prefix, above_count;
  __shared__ unsigned histogram_scan[32];
  __shared__ unsigned count_scan[32];
  __shared__ unsigned above_mass;
  __shared__ float compensation;
  __shared__ unsigned total_weight;
  __shared__ unsigned random_weight;
  __shared__ int selected, owner;
  __shared__ unsigned residual;

  if constexpr (!Cooperative) {
    float maximum = -FLT_MAX;

    for (int i = t; i < V / 4; i += blockDim.x) {
      float4 x = reinterpret_cast<const float4*>(logits)[i];
      float4 penalty;
      if constexpr (Penalties) penalty = reinterpret_cast<const float4*>(penalties)[i];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        float v = safe_float(reinterpret_cast<float*>(&x)[j]);
        if constexpr (Penalties) v -= reinterpret_cast<float*>(&penalty)[j];
        v = safe_float(v * inv_temperature);
        reinterpret_cast<float*>(&x)[j] = v;
        maximum = fmaxf(maximum, v);
      }
      reinterpret_cast<float4*>(probs)[i] = x;
    }

    maximum = block_reduce<float, true>(maximum, fscratch);
    float denominator = 0.f;

    for (int i = t; i < V / 4; i += blockDim.x) {
      float4 x = reinterpret_cast<const float4*>(probs)[i];
      float sum = 0.f;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        float p = exp2f(reinterpret_cast<float*>(&x)[j] - maximum);
        reinterpret_cast<float*>(&x)[j] = p;
        sum += p;
      }
      reinterpret_cast<float4*>(probs)[i] = x;
      denominator += sum;
    }

    denominator = block_reduce(denominator, fscratch);
    const float inv_denominator = 1.f / denominator;

    for (int i = t; i < V / 4; i += blockDim.x) {
      float4 x = reinterpret_cast<const float4*>(probs)[i];
#pragma unroll
      for (int j = 0; j < 4; ++j) reinterpret_cast<float*>(&x)[j] *= inv_denominator;
      reinterpret_cast<float4*>(probs)[i] = x;
    }
  }
  if (t == 0) {
    prefix = 0;
    above_mass = 0;
    above_count = 0;
    compensation = 1.f;
  }
  __syncthreads();

  // Nonnegative IEEE float bits are ordered like their values. Refine only
  // the boundary bucket with the requested bit partition.
  if (top_k < V || top_p < 1.) {
    for (int level = 0; level < layout.passes; ++level) {
      const int shift = layout.shifts[level], bins = 1 << layout.widths[level];
      unsigned overflow = 1;
      if constexpr (CachedCandidates > 0) {
        if (level >= 2) overflow = block_reduce(candidate_overflow, iscratch);
      }
      for (int bin = t; bin < bins; bin += blockDim.x) {
        for (int shard = 0; shard < layout.shards; ++shard) {
          const int index = shard * histogram_stride + bin;
          mass[index] = 0;
          counts[index] = 0;
        }
      }
      __syncthreads();
      const unsigned boundary = prefix;
      const int shard = (t >> 5) % layout.shards;
      if (Cooperative && level == 0) {
        const int tiles = (V + 2047) / 2048;
        const int stored_bins = bins / 2;
        const unsigned* h = first_histogram + int64_t(b) * tiles * stored_bins * 2;
        for (int bin = t; bin < bins; bin += blockDim.x) {
          unsigned sum = 0;
          unsigned count = 0;
          for (int tile = 0; bin < stored_bins && tile < tiles; ++tile) {
            uint2 value = reinterpret_cast<const uint2*>(h)[tile * stored_bins + bin];
            sum += value.x;
            count += value.y;
          }
          mass[bin] = sum;
          counts[bin] = count;
        }
      } else {
        auto insert = [&](float p) {
          unsigned bits = __float_as_uint(p);
          if (level == 0 || (bits >> (shift + layout.widths[level])) ==
                                (boundary >> (shift + layout.widths[level]))) {
            if constexpr (CachedCandidates > 0) {
              if (level == 1) {
                if (candidate_count < CachedCandidates)
                  candidate_cache[candidate_count * blockDim.x + t] = p;
                else
                  candidate_overflow = 1;
                ++candidate_count;
              }
            }
            int bin = shard * histogram_stride + ((bits >> shift) & (bins - 1));
            unsigned q;
            q = __float2uint_rn(p * mass_scale);
            atomicAdd(mass + bin, q);
            atomicAdd(counts + bin, 1u);
          }
        };
        if (CachedCandidates > 0 && level >= 2 && !overflow) {
          for (int i = 0; i < int(candidate_count); ++i)
            insert(candidate_cache[i * blockDim.x + t]);
        } else {
          for (int i = t; i < V / 4; i += blockDim.x) {
            float4 x = reinterpret_cast<const float4*>(probs)[i];
            insert(x.x);
            insert(x.y);
            insert(x.z);
            insert(x.w);
          }
        }
      }
      __syncthreads();
      // Scan contiguous descending bin chunks in parallel. Only the
      // winning chunk (at most ceil(bins/blockDim.x) bins) is searched
      // serially; no thread walks the entire histogram.
      const unsigned previous_mass = above_mass;
      const unsigned previous_count = above_count;
      const int bin_chunk = (bins + int(blockDim.x) - 1) / int(blockDim.x);
      const int begin_bin = t * bin_chunk;
      const int end_bin = min(begin_bin + bin_chunk, bins);
      unsigned chunk_mass = 0;
      unsigned chunk_count = 0;
      for (int position = begin_bin; position < end_bin; ++position) {
        const int bin = bins - 1 - position;
        for (int shard = 0; shard < layout.shards; ++shard) {
          const unsigned value = mass[shard * histogram_stride + bin];
          chunk_mass += value;
          chunk_count += counts[shard * histogram_stride + bin];
        }
      }
      const unsigned cumulative_mass = block_scan(chunk_mass, histogram_scan);
      const unsigned cumulative_count = block_scan(chunk_count, count_scan);
      // top_p=1 disables the mass bound, even when rounded softmax
      // probabilities sum slightly above one. Preserve rare top-k weights.
      const unsigned raw_target =
          top_p >= 1.f ? ~unsigned(0) : static_cast<unsigned>(top_p * mass_scale);
      const unsigned target = raw_target > 0 ? raw_target : unsigned(1);
      unsigned before_mass = previous_mass + cumulative_mass - chunk_mass;
      unsigned before_count = previous_count + cumulative_count - chunk_count;
      if (chunk_count && before_count < unsigned(top_k) && before_mass < target &&
          (previous_count + cumulative_count >= unsigned(top_k) ||
           previous_mass + cumulative_mass >= target)) {
        for (int position = begin_bin; position < end_bin; ++position) {
          const int bin = bins - 1 - position;
          unsigned m = 0;
          unsigned c = 0;
          for (int shard = 0; shard < layout.shards; ++shard) {
            const unsigned value = mass[shard * histogram_stride + bin];
            m += value;
            c += counts[shard * histogram_stride + bin];
          }
          if (c && (before_count + c >= unsigned(top_k) || before_mass + m >= target)) {
            prefix |= unsigned(bin) << shift;
            above_mass = before_mass;
            above_count = before_count;
            if (level == layout.passes - 1) {
              const float by_k = float(top_k - int(before_count)) / float(c);
              const unsigned remaining = raw_target > before_mass ? raw_target - before_mass : 0;
              const float by_p = m > 0 ? float(remaining) / float(m) : 1.f;
              compensation = fmaxf(0.f, fminf(1.f, fminf(by_k, by_p)));
              if (before_mass == 0 && compensation == 0.f && top_p > 0.f) compensation = FLT_MIN;
            }
            break;
          }
          before_mass += m;
          before_count += c;
        }
      }
      __syncthreads();
    }
  }
  const float threshold = __uint_as_float(prefix);
  if constexpr (Cooperative) {
    if (t == 0) {
      unsigned* control = const_cast<unsigned*>(first_histogram);
      control[0] = __float_as_uint(threshold);
      control[1] = __float_as_uint(compensation);
    }
    return;
  }
  float retained = 0.f;

  for (int i = t; i < V / 4; i += blockDim.x) {
    float4 x = reinterpret_cast<const float4*>(probs)[i];
    float sum = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float p = reinterpret_cast<float*>(&x)[j];
      sum += p > threshold ? p : (p == threshold ? p * compensation : 0.f);
    }
    retained += sum;
  }

  retained = block_reduce(retained, fscratch);
  // Normalize with headroom below UINT32_MAX before the integer CDF scan.
  const float factor = retained > 0.f ? 4.26e9f / retained : 0.f;
  unsigned local = 0;

  for (int i = t; i < V / 4; i += blockDim.x) {
    float4 x = reinterpret_cast<const float4*>(probs)[i];
    unsigned values[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float p = reinterpret_cast<float*>(&x)[j];
      const float filtered = p > threshold ? p : (p == threshold ? p * compensation : 0.f);
      unsigned weight;

      weight = unsigned(isfinite(factor) ? filtered * factor : (filtered / retained) * 4.26e9f);

      values[j] = weight;
      local += weight;
    }

    reinterpret_cast<uint4*>(weights)[i] = make_uint4(values[0], values[1], values[2], values[3]);
  }

  unsigned cumulative = block_scan(local, iscratch);
  if (t == blockDim.x - 1) total_weight = cumulative;
  __syncthreads();
  if (t == 0) {
    selected = 0;
    owner = 0;
    residual = 0;

    random_weight = total_weight > 0 ? random_below32(states, total_weight) : 0;
  }
  __syncthreads();
  // First select the thread's strided token group using an exact integer interval.
  if (total_weight > 0 && random_weight >= cumulative - local && random_weight < cumulative) {
    owner = t;
    residual = random_weight - (cumulative - local);
  }
  __syncthreads();
  // Distribute the winning group's tokens over the block instead of making
  // one thread serially read V/blockDim.x weights. Integer chunk scans preserve
  // the same categorical interval and skip zero-weight tokens.
  const int group_size = 4 * ((V / 4 - owner + int(blockDim.x) - 1) / int(blockDim.x));
  const int chunk_size = (group_size + int(blockDim.x) - 1) / int(blockDim.x);
  const int begin = t * chunk_size, end = min(begin + chunk_size, group_size);
  unsigned chunk_weight = 0;
  for (int j = begin; j < end; ++j)
    chunk_weight += weights[4 * (owner + (j / 4) * blockDim.x) + j % 4];
  unsigned chunk_prefix = block_scan(chunk_weight, iscratch);
  if (total_weight > 0 && residual >= chunk_prefix - chunk_weight && residual < chunk_prefix) {
    unsigned r = residual - (chunk_prefix - chunk_weight);
    for (int j = begin; j < end; ++j) {
      const int i = 4 * (owner + (j / 4) * blockDim.x) + j % 4;
      if (r < weights[i]) {
        selected = i;
        break;
      }
      r -= weights[i];
    }
  }
  __syncthreads();
  if (t == 0) outputs[b] = selected;
  if constexpr (Penalties) {
    for (int i = t; i < V; i += blockDim.x) {
      float p = penalties[i];
      const bool seen = p != 0.f || signbit(p);
      const bool sampled = i == selected;
      p = fmaf(p, decay, sampled ? repetition + (seen ? 0.f : presence) : 0.f);
      penalties[i] = (seen || sampled) && p == 0.f ? -0.f : p;
    }
  }
}

__device__ __forceinline__ bool sampling_row_is_active(int row, const int* slot_indices,
                                                       int64_t num_slots,
                                                       const int* num_active_samples,
                                                       const int* status) {
  if (status != nullptr) return status[0] == 0 && row < status[1];
  // A single-row call cannot contain duplicate slots. Check it at the consumer.
  const int active = num_active_samples == nullptr ? 1 : num_active_samples[0];
  return active == 1 && slot_indices[0] >= 0 && slot_indices[0] < num_slots;
}

template <bool Penalties>
__global__ void __launch_bounds__(1024, 1)
    sampling_kernel(int V, const float* logits, float* penalties, int* outputs, RAND* states,
                    float* probs, unsigned* weights, SamplingControls controls,
                    HistogramLayout layout, const int* slot_indices, int64_t num_slots,
                    const int* num_active_samples, const int* status) {
  const int row = blockIdx.x;
  if (!sampling_row_is_active(row, slot_indices, num_slots, num_active_samples, status)) {
    if (threadIdx.x == 0) outputs[row] = -1;
    return;
  }
  controls = sampling_row_controls(controls, row, V);
  sampling_body<Penalties>(V, logits, penalties, outputs, states, probs, weights,
                           1.4426950408889634f / controls.temperature, controls.top_k,
                           controls.top_p, controls.presence, controls.frequency, controls.decay,
                           layout, nullptr, slot_indices);
}

// B=1,V=65536 cooperative fast path. Every exponential uses one global
// maximum, so equal logits retain equal probabilities across tiles.
__device__ __forceinline__ void cooperative_max(int V, const float* logits, float* probs,
                                                float* partials, float inv_temperature) {
  const int t = threadIdx.x, tile = blockIdx.x, begin = tile * 2048, end = min(begin + 2048, V);
  __shared__ float scratch[32];
  float maximum = -FLT_MAX;
  for (int i = begin + 4 * t; i < end; i += 4 * blockDim.x) {
    float4 x = reinterpret_cast<const float4*>(logits)[i / 4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float v = safe_float(safe_float(reinterpret_cast<float*>(&x)[j]) * inv_temperature);
      maximum = fmaxf(maximum, v);
      reinterpret_cast<float*>(&x)[j] = v;
    }
    reinterpret_cast<float4*>(probs)[i / 4] = x;
  }
  maximum = block_reduce<float, true>(maximum, scratch);
  if (t == 0) partials[2 * tile] = maximum;
}
__device__ __forceinline__ void global_exp_body(int V, float* probs, float* partials) {
  const int t = threadIdx.x, tile = blockIdx.x, tiles = (V + 2047) / 2048;
  __shared__ float scratch[32];
  float maximum = t < tiles ? partials[2 * t] : -FLT_MAX;
  maximum = block_reduce<float, true>(maximum, scratch);
  const int begin = tile * 2048, end = min(begin + 2048, V);
  float sum = 0.f;
  for (int i = begin + 4 * t; i < end; i += 4 * blockDim.x) {
    float4 x = reinterpret_cast<float4*>(probs)[i / 4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float p = exp2f(reinterpret_cast<float*>(&x)[j] - maximum);
      sum += p;
      reinterpret_cast<float*>(&x)[j] = p;
    }
    reinterpret_cast<float4*>(probs)[i / 4] = x;
  }
  sum = block_reduce(sum, scratch);
  if (t == 0) partials[2 * tile + 1] = sum;
}

__device__ __forceinline__ void cooperative_normalize_histogram(int V, float* probs,
                                                                const float* partials,
                                                                unsigned* histograms, int width,
                                                                int shift, bool needed) {
  const int t = threadIdx.x, tile = blockIdx.x, tiles = (V + 2047) / 2048, bins = 1 << (width - 1);
  __shared__ float scratch[32];
  extern __shared__ unsigned storage[];
  unsigned* mass = storage;
  unsigned* counts = storage + bins;
  const float denominator = block_reduce(t < tiles ? partials[2 * t + 1] : 0.f, scratch);
  const float scale = 1.f / denominator;
  if (needed)
    for (int bin = t; bin < bins; bin += blockDim.x) {
      mass[bin] = 0;
      counts[bin] = 0;
    }
  __syncthreads();
  const int begin = tile * 2048, end = min(begin + 2048, V);
  for (int i = begin + 4 * t; i < end; i += 4 * blockDim.x) {
    float4 x = reinterpret_cast<float4*>(probs)[i / 4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float p = reinterpret_cast<float*>(&x)[j] * scale;
      reinterpret_cast<float*>(&x)[j] = p;
      if (needed) {
        int bin = __float_as_uint(p) >> shift;
        atomicAdd(mass + bin, __float2uint_rn(p * 0x1p31f));
        atomicAdd(counts + bin, 1u);
      }
    }
    reinterpret_cast<float4*>(probs)[i / 4] = x;
  }
  __syncthreads();
  if (needed) {
    unsigned* h = histograms + int64_t(tile) * bins * 2;
    for (int bin = t; bin < bins; bin += blockDim.x) {
      h[2 * bin] = mass[bin];
      h[2 * bin + 1] = counts[bin];
    }
  }
}
__global__ void __launch_bounds__(1024, 1)
    sampling_cooperative_kernel(int V, const float* logits, int* outputs, RAND* states,
                                float* probs, unsigned* weights, float* partials,
                                unsigned* histograms, HistogramLayout layout,
                                const int* slot_indices, int64_t num_slots,
                                const int* num_active_samples, SamplingControls controls) {
  if (!sampling_row_is_active(0, slot_indices, num_slots, num_active_samples, nullptr)) {
    if (blockIdx.x == 0 && threadIdx.x == 0) outputs[0] = -1;
    return;
  }
  controls = sampling_row_controls(controls, 0, V);
  const float inv_temperature = 1.4426950408889634f / controls.temperature;
  const int k = controls.top_k;
  const float p = controls.top_p;
  states += slot_indices[0];
  // All tiles share a maximum and denominator, preserving ties across tiles.
  cooperative_max(V, logits, probs, partials, inv_temperature);
  cooperative_groups::this_grid().sync();
  global_exp_body(V, probs, partials);
  cooperative_groups::this_grid().sync();
  cooperative_normalize_histogram(V, probs, partials, histograms, layout.widths[0],
                                  layout.shifts[0], k < V || p < 1.f);
  cooperative_groups::this_grid().sync();
  // Refine the histogram in block zero and publish the cutoff and tie fraction.
  if (blockIdx.x == 0) {
    sampling_body<false, true, 8>(V, logits, nullptr, outputs, states, probs, weights,
                                  inv_temperature, k, p, 0.f, 0.f, 0.f, layout, histograms);
  }
  cooperative_groups::this_grid().sync();
  // Apply the cutoff, then quantize using the mass retained by the whole grid.
  const int t = threadIdx.x, tile = blockIdx.x, tiles = 32;
  const float threshold = __uint_as_float(histograms[0]),
              compensation = __uint_as_float(histograms[1]);
  const int i = tile * 512 + t;
  float4 x = make_float4(0.f, 0.f, 0.f, 0.f);
  if (t < 512) {
    x = reinterpret_cast<float4*>(probs)[i];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float p = reinterpret_cast<float*>(&x)[j];
      reinterpret_cast<float*>(&x)[j] =
          p > threshold ? p : (p == threshold ? p * compensation : 0.f);
    }
  }
  __shared__ float fscratch[32];
  __shared__ unsigned iscratch[32];
  float retained = block_reduce(x.x + x.y + x.z + x.w, fscratch);
  if (t == 0) partials[tile] = retained;
  cooperative_groups::this_grid().sync();
  retained = block_reduce(t < tiles ? partials[t] : 0.f, fscratch);
  const float factor = retained > 0.f ? 4.26e9f / retained : 0.f;
  uint4 w = make_uint4(0, 0, 0, 0);
  if (t < 512) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float value = reinterpret_cast<float*>(&x)[j];
      reinterpret_cast<unsigned*>(&w)[j] =
          unsigned(isfinite(factor) ? value * factor : (value / retained) * 4.26e9f);
    }
    reinterpret_cast<uint4*>(weights)[i] = w;
  }
  unsigned tile_weight = block_reduce(w.x + w.y + w.z + w.w, iscratch);
  if (t == 0) histograms[2 + tile] = tile_weight;
  cooperative_groups::this_grid().sync();
  // Select a tile by integer mass, then select a token within that tile.
  if (tile == 0) {
    __shared__ unsigned total, random, residual, chosen;
    unsigned local = t < tiles ? histograms[2 + t] : 0;
    unsigned cumulative = block_scan(local, iscratch);
    if (t == blockDim.x - 1) total = cumulative;
    __syncthreads();
    if (!total) {
      if (t == 0) outputs[0] = 0;
      return;
    }
    if (t == 0) random = random_below32(states, total);
    __syncthreads();
    if (local && random >= cumulative - local && random < cumulative) {
      chosen = t;
      residual = random - (cumulative - local);
    }
    __syncthreads();
    w = t < 512 ? reinterpret_cast<uint4*>(weights)[chosen * 512 + t] : make_uint4(0, 0, 0, 0);
    local = w.x + w.y + w.z + w.w;
    cumulative = block_scan(local, iscratch);
    if (local && residual >= cumulative - local && residual < cumulative) {
      unsigned r = residual - (cumulative - local);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        unsigned q = reinterpret_cast<unsigned*>(&w)[j];
        if (r < q) {
          outputs[0] = chosen * 2048 + 4 * t + j;
          break;
        }
        r -= q;
      }
    }
  }
}

}  // namespace

int64_t sampling_state_size_cuda() { return static_cast<int64_t>(sizeof(RAND)); }

at::Tensor setup_sampling_states_cuda(int64_t seed, int64_t num_slots) {
  auto states = at::empty({num_slots, sampling_state_size_cuda()},
                          at::TensorOptions().dtype(at::kChar).device(at::kCUDA));
  create_rng_kernel<<<(num_slots + RAND_SETUP_THREADS - 1) / RAND_SETUP_THREADS, RAND_SETUP_THREADS,
                      0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<RAND*>(states.data_ptr()), static_cast<unsigned long long>(seed), num_slots);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return states;
}

namespace {

__global__ void validate_sampling_metadata_kernel(const int* __restrict__ slot_indices,
                                                  const int* __restrict__ num_active_samples,
                                                  int sample_capacity, int64_t num_slots,
                                                  int64_t table_capacity, int* __restrict__ status,
                                                  int* __restrict__ seen_slots) {
  __shared__ int active_samples;
  __shared__ int error;
  if (threadIdx.x == 0) {
    active_samples = num_active_samples == nullptr ? sample_capacity : num_active_samples[0];
    error = active_samples < 0 || active_samples > sample_capacity ? 1 : 0;
    if (error != 0) {
      active_samples = 0;
    }
  }
  __syncthreads();
  if (active_samples <= 1) {
    if (threadIdx.x == 0) {
      if (active_samples == 1 && (slot_indices[0] < 0 || slot_indices[0] >= num_slots)) error = 1;
      status[0] = error;
      status[1] = error == 0 ? active_samples : 0;
    }
    return;
  }
  for (int64_t index = threadIdx.x; index < table_capacity; index += blockDim.x) {
    seen_slots[index] = -1;
  }
  __syncthreads();
  for (int row = static_cast<int>(threadIdx.x); row < active_samples;
       row += static_cast<int>(blockDim.x)) {
    const int slot = slot_indices[row];
    if (slot < 0 || slot >= num_slots) {
      atomicExch(&error, 1);
      continue;
    }
    const unsigned mask = static_cast<unsigned>(table_capacity - 1);
    unsigned index = (static_cast<unsigned>(slot) * 2654435761u) >> __clz(mask);
    while (true) {
      const int previous = atomicCAS(seen_slots + index, -1, slot);
      if (previous == -1) break;
      if (previous == slot) {
        atomicExch(&error, 1);
        break;
      }
      index = (index + 1) & mask;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    status[0] = error;
    status[1] = error == 0 ? active_samples : 0;
  }
}

struct SamplingMetadata {
  at::Tensor status;
  at::Tensor workspace;
};

SamplingMetadata prepare_sampling_metadata(const at::Tensor& logits, const at::Tensor& states,
                                           const at::Tensor& slot_indices,
                                           const at::Tensor& num_active_samples) {
  auto status = at::empty({2}, slot_indices.options());
  int64_t table_capacity = 1;
  while (table_capacity < 2 * logits.size(0)) table_capacity <<= 1;
  auto workspace = at::empty({table_capacity}, slot_indices.options());
  validate_sampling_metadata_kernel<<<1, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      slot_indices.data_ptr<int>(),
      num_active_samples.defined() ? num_active_samples.data_ptr<int>() : nullptr,
      static_cast<int>(logits.size(0)), states.size(0), table_capacity, status.data_ptr<int>(),
      workspace.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return SamplingMetadata{std::move(status), std::move(workspace)};
}

constexpr int SAMPLING_THREADS = 1024;
constexpr int SAMPLING_TILES = 32;
constexpr size_t HISTOGRAM_BYTES = 2 * 2049 * 2 * sizeof(unsigned);
constexpr size_t COOPERATIVE_BYTES = HISTOGRAM_BYTES + 8 * SAMPLING_THREADS * sizeof(float);

bool sampling_cooperative_available(int device) {
  static thread_local int configured_device = -1;
  static thread_local bool available = false;
  if (configured_device != device) {
    const auto* properties = at::cuda::getDeviceProperties(device);
    cudaFuncAttributes attributes{};
    C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes, sampling_cooperative_kernel));
    available = properties->cooperativeLaunch && properties->sharedMemPerBlockOptin >=
                                                     COOPERATIVE_BYTES + attributes.sharedSizeBytes;
    if (available) {
      C10_CUDA_CHECK(cudaFuncSetAttribute(sampling_cooperative_kernel,
                                          cudaFuncAttributeMaxDynamicSharedMemorySize,
                                          COOPERATIVE_BYTES));
      int active_blocks;
      C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &active_blocks, sampling_cooperative_kernel, SAMPLING_THREADS, COOPERATIVE_BYTES));
      available = active_blocks * properties->multiProcessorCount >= SAMPLING_TILES;
    }
    configured_device = device;
  }
  return available;
}

template <bool Penalties>
at::Tensor sampling_forward(at::Tensor logits, at::Tensor penalties, at::Tensor states,
                            at::Tensor slot_indices, at::Tensor num_active_samples,
                            SamplingControls controls) {
  const int B = static_cast<int>(logits.size(0));
  int V = static_cast<int>(logits.size(1));
  HistogramLayout layout{3, 2048, {10, 10, 11}, {21, 11, 0}, 2};
  auto probs = at::empty(logits.sizes(), logits.options());
  auto weights = at::empty(logits.sizes(), logits.options().dtype(at::kInt));
  auto output = at::empty({B}, logits.options().dtype(at::kInt));
  SamplingMetadata metadata;
  if (B > 1) metadata = prepare_sampling_metadata(logits, states, slot_indices, num_active_samples);
  auto stream = at::cuda::getCurrentCUDAStream();
  const float* l = logits.data_ptr<float>();
  float* prob = probs.data_ptr<float>();
  unsigned* w = reinterpret_cast<unsigned*>(weights.data_ptr<int>());
  RAND* rng = reinterpret_cast<RAND*>(states.data_ptr());
  int* out = output.data_ptr<int>();
  int64_t num_slots = states.size(0);
  const int* slots = slot_indices.data_ptr<int>();
  const int* status = metadata.status.defined() ? metadata.status.data_ptr<int>() : nullptr;
  const int* active = num_active_samples.defined() ? num_active_samples.data_ptr<int>() : nullptr;
  if constexpr (!Penalties) {
    if (B == 1 && V == 65536 && sampling_cooperative_available(logits.get_device())) {
      auto partials = at::empty({2 * SAMPLING_TILES}, logits.options());
      auto histograms = at::empty({SAMPLING_TILES * 512 * 2}, logits.options().dtype(at::kInt));
      float* part = partials.data_ptr<float>();
      unsigned* hist = reinterpret_cast<unsigned*>(histograms.data_ptr<int>());
      void* args[] = {&V,    &l,      &out,   &rng,       &prob,   &w,       &part,
                      &hist, &layout, &slots, &num_slots, &active, &controls};
      C10_CUDA_CHECK(cudaLaunchCooperativeKernel(
          reinterpret_cast<void*>(sampling_cooperative_kernel), dim3(SAMPLING_TILES),
          dim3(SAMPLING_THREADS), args, COOPERATIVE_BYTES, stream));
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      return output;
    }
  }
  static thread_local int configured_device = -1;
  if (configured_device != logits.get_device()) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        sampling_kernel<Penalties>, cudaFuncAttributeMaxDynamicSharedMemorySize, HISTOGRAM_BYTES));
    configured_device = logits.get_device();
  }
  sampling_kernel<Penalties><<<B, SAMPLING_THREADS, HISTOGRAM_BYTES, stream>>>(
      V, l, Penalties ? penalties.data_ptr<float>() : nullptr, out, rng, prob, w, controls, layout,
      slots, num_slots, active, status);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

SamplingControls sampling_scalar_controls(double presence, double frequency, double decay,
                                          double temperature, int64_t top_k, double top_p, int V) {
  SamplingControls controls;
  controls.presence = static_cast<float>(presence);
  controls.frequency = static_cast<float>(frequency);
  controls.decay = static_cast<float>(decay);
  controls.temperature = static_cast<float>(fmin(fmax(temperature, .001), 1000.));
  controls.top_k = top_k <= 0 || top_k > V ? V : static_cast<int>(top_k);
  controls.top_p = static_cast<float>(fmin(fmax(top_p, 0.), 1.));
  // Preserve a positive top-p that underflows during host double-to-float conversion.
  if (top_p > 0. && controls.top_p == 0.f) controls.top_p = FLT_MIN;
  return controls;
}

}  // namespace

at::Tensor sampling_temperature_topk_topp_scalar_cuda(at::Tensor logits, at::Tensor states,
                                                      at::Tensor slot_indices, double temperature,
                                                      int64_t top_k, double top_p,
                                                      at::Tensor num_active_samples) {
  auto controls = sampling_scalar_controls(0., 0., 1., temperature, top_k, top_p,
                                           static_cast<int>(logits.size(1)));
  return sampling_forward<false>(logits, {}, states, slot_indices, num_active_samples, controls);
}

at::Tensor sampling_temperature_topk_topp_per_request_cuda(at::Tensor logits, at::Tensor states,
                                                           at::Tensor slot_indices,
                                                           at::Tensor temperatures,
                                                           at::Tensor top_ks, at::Tensor top_ps,
                                                           at::Tensor num_active_samples) {
  SamplingControls controls;
  controls.temperatures = temperatures.data_ptr<float>();
  controls.top_ks = top_ks.data_ptr<int>();
  controls.top_ps = top_ps.data_ptr<float>();
  return sampling_forward<false>(logits, {}, states, slot_indices, num_active_samples, controls);
}

at::Tensor sampling_six_parameter_scalar_cuda(at::Tensor logits, at::Tensor penalties,
                                              at::Tensor states, at::Tensor slot_indices,
                                              double presence_penalty, double frequency_penalty,
                                              double penalty_decay, double temperature,
                                              int64_t top_k, double top_p,
                                              at::Tensor num_active_samples) {
  auto controls =
      sampling_scalar_controls(presence_penalty, frequency_penalty, penalty_decay, temperature,
                               top_k, top_p, static_cast<int>(logits.size(1)));
  return sampling_forward<true>(logits, penalties, states, slot_indices, num_active_samples,
                                controls);
}

at::Tensor sampling_six_parameter_per_request_cuda(
    at::Tensor logits, at::Tensor penalties, at::Tensor states, at::Tensor slot_indices,
    at::Tensor presence_penalties, at::Tensor frequency_penalties, at::Tensor penalty_decays,
    at::Tensor temperatures, at::Tensor top_ks, at::Tensor top_ps, at::Tensor num_active_samples) {
  SamplingControls controls;
  controls.presences = presence_penalties.data_ptr<float>();
  controls.frequencies = frequency_penalties.data_ptr<float>();
  controls.decays = penalty_decays.data_ptr<float>();
  controls.temperatures = temperatures.data_ptr<float>();
  controls.top_ks = top_ks.data_ptr<int>();
  controls.top_ps = top_ps.data_ptr<float>();
  return sampling_forward<true>(logits, penalties, states, slot_indices, num_active_samples,
                                controls);
}
