// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the Albatross project
// SPDX-FileCopyrightText: Copyright contributors to the FlashRWKV2 project
//
// Source: BlinkDL/Albatross/faster3a_2607/cuda/rwkv7_fast_ops_fp16.cu,
// revision ee3308f6922e59f2166c7fac3c5a192340a2b48e.
//
// The complete CMix owner uses this TokenShift arithmetic and its upstream launch families.
// mechanically.  The only adaptation is the packed request address layer:
// cu_seqlens identifies the request boundary and state_indices identifies the
// request's shift-state slot.  No alternate CMix implementation is kept here.

#include <cuda_fp16.h>
#include "validation.h"

#include <cstdint>
#include <vector>

namespace {

constexpr unsigned int kMaxGridDimYZ = 65535;

using dtype = torch::headeronly::Half;

__device__ inline __half2 load_h2(const dtype* ptr) {
  return *reinterpret_cast<const __half2*>(ptr);
}

__device__ inline void store_h2(dtype* ptr, float x0, float x1) {
  *reinterpret_cast<__half2*>(ptr) = __floats2half2_rn(x0, x1);
}

template <bool Grid3D, bool UpdateShift>
__global__ void cmix_tokenshift_varlen_kernel(
    int batch_size,
    int total_tokens,
    int C,
    const dtype* __restrict__ x,
    dtype* __restrict__ shift_state,
    const dtype* __restrict__ x_k,
    dtype* __restrict__ out,
    const int* __restrict__ token_predecessor,
    const int* __restrict__ metadata_status) {
  const int c_pairs = C >> 1;
  const int64_t pair_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int pair = static_cast<int>(pair_idx);
  const int token = Grid3D
      ? static_cast<int>(blockIdx.y)
      : static_cast<int>(pair_idx / c_pairs);
  const int c_pair = Grid3D
      ? pair
      : static_cast<int>(pair_idx - static_cast<int64_t>(token) * c_pairs);
  if (c_pair >= c_pairs || token >= total_tokens) {
    return;
  }

  const int c = c_pair << 1;
  const int64_t idx = static_cast<int64_t>(token) * C + c;
  if (metadata_status[0] != 0) {
    store_h2(out + idx, __int_as_float(0x7fffffff), __int_as_float(0x7fffffff));
    return;
  }
  if (token >= metadata_status[1]) {
    return;
  }

  const int predecessor = token_predecessor[token];
  const bool sequence_start = predecessor < 0;
  const int slot = -predecessor - 1;
  const int64_t previous_idx = sequence_start
      ? static_cast<int64_t>(slot) * C + c
      : static_cast<int64_t>(predecessor) * C + c;

  // This is the canonical Albatross cmix_tokenshift arithmetic.
  const __half2 cur2 = load_h2(x + idx);
  const __half2 prev2 = sequence_start
      ? load_h2(shift_state + previous_idx)
      : load_h2(x + previous_idx);
  const float2 cur = __half22float2(cur2);
  const float2 prev = __half22float2(prev2);
  const float2 mix = __half22float2(load_h2(x_k + c));
  store_h2(
      out + idx,
      cur.x + (prev.x - cur.x) * mix.x,
      cur.y + (prev.y - cur.y) * mix.y);

  if constexpr (UpdateShift) {
    // This specialization is used only when every request has one token.
    *reinterpret_cast<__half2*>(shift_state + static_cast<int64_t>(slot) * C + c) = cur2;
  }
}

__global__ void update_shift_state_last_varlen_kernel(
    int batch_size,
    int C,
    const dtype* __restrict__ x,
    dtype* __restrict__ shift_state,
    const int* __restrict__ cu_seqlens,
    const int* __restrict__ state_indices,
    const int* __restrict__ metadata_status) {
  const int sequence = static_cast<int>(blockIdx.x);
  if (sequence >= batch_size || metadata_status[0] != 0 ||
      sequence >= metadata_status[2]) {
    return;
  }
  const int token = cu_seqlens[sequence + 1] - 1;
  const int slot = state_indices[sequence];
  const int pairs = C >> 1;
  for (int c_pair = threadIdx.x; c_pair < pairs; c_pair += blockDim.x) {
    const int c = c_pair << 1;
    const int64_t src_idx = static_cast<int64_t>(token) * C + c;
    *reinterpret_cast<__half2*>(shift_state + static_cast<int64_t>(slot) * C + c) =
        load_h2(x + src_idx);
  }
}

template <int Threads>
__device__ __forceinline__ float block_sum(float value) {
  __shared__ float partial[Threads / 32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  if (lane == 0) partial[warp] = value;
  __syncthreads();
  value = threadIdx.x < Threads / 32 ? partial[lane] : 0.0f;
  if (warp == 0) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      value += __shfl_down_sync(0xffffffffu, value, offset);
    }
  }
  if (threadIdx.x == 0) partial[0] = value;
  __syncthreads();
  const float result = partial[0];
  __syncthreads();
  return result;
}

template <int Threads>
__global__ __launch_bounds__(Threads, 1) void res_ln_cmix_tokenshift_fused_kernel(
    const dtype* __restrict__ x,
    const dtype* __restrict__ res,
    dtype* __restrict__ shift_state,
    const dtype* __restrict__ weight,
    const dtype* __restrict__ bias,
    const dtype* __restrict__ x_k,
    dtype* __restrict__ res_out,
    dtype* __restrict__ mixed,
    const int* __restrict__ token_predecessor,
    const int* __restrict__ metadata_status,
    int64_t rows,
    float eps) {
  constexpr int C = 4096;
  constexpr int pairs = C / 2;
  const int64_t row = blockIdx.x;
  if (row >= rows) return;
  const int64_t base = row * C;
  const int64_t base2 = row * pairs;
  if (metadata_status[0] != 0) {
    const __half2 invalid = __floats2half2_rn(
        __int_as_float(0x7fffffff), __int_as_float(0x7fffffff));
    for (int p = threadIdx.x; p < pairs; p += Threads) {
      reinterpret_cast<__half2*>(res_out)[base2 + p] = invalid;
      reinterpret_cast<__half2*>(mixed)[base2 + p] = invalid;
    }
    return;
  }
  // Captured capacity may exceed the active request count on graph replay.
  if (row >= metadata_status[2]) return;
  const int slot = -token_predecessor[row] - 1;
  float sum = 0.0f;
#pragma unroll
  for (int k = 0; k < C / Threads; ++k) {
    const int c = threadIdx.x + k * Threads;
    sum += __half2float(*reinterpret_cast<const __half*>(x + base + c)) +
           __half2float(*reinterpret_cast<const __half*>(res + base + c));
  }
  const float mean = block_sum<Threads>(sum) / C;
  float var = 0.0f;
#pragma unroll
  for (int k = 0; k < C / Threads; ++k) {
    const int c = threadIdx.x + k * Threads;
    const float value =
        __half2float(*reinterpret_cast<const __half*>(x + base + c)) +
        __half2float(*reinterpret_cast<const __half*>(res + base + c));
    const float delta = value - mean;
    var += delta * delta;
  }
  const float rstd = rsqrtf(block_sum<Threads>(var) / C + eps);
#pragma unroll
  for (int p = threadIdx.x; p < pairs; p += Threads) {
    const float2 xv = __half22float2(reinterpret_cast<const __half2*>(x)[base2 + p]);
    const float2 rv = __half22float2(reinterpret_cast<const __half2*>(res)[base2 + p]);
    const float2 w = __half22float2(reinterpret_cast<const __half2*>(weight)[p]);
    const float2 b = __half22float2(reinterpret_cast<const __half2*>(bias)[p]);
    const float2 mix = __half22float2(reinterpret_cast<const __half2*>(x_k)[p]);
    const float2 prev = __half22float2(
        reinterpret_cast<const __half2*>(shift_state)[static_cast<int64_t>(slot) * pairs + p]);
    const float s0 = xv.x + rv.x;
    const float s1 = xv.y + rv.y;
    const __half2 norm2 = __floats2half2_rn(
        (s0 - mean) * rstd * w.x + b.x,
        (s1 - mean) * rstd * w.y + b.y);
    const float2 norm = __half22float2(norm2);
    reinterpret_cast<__half2*>(res_out)[base2 + p] = __floats2half2_rn(s0, s1);
    reinterpret_cast<__half2*>(mixed)[base2 + p] = __floats2half2_rn(
        norm.x + (prev.x - norm.x) * mix.x,
        norm.y + (prev.y - norm.y) * mix.y);
    reinterpret_cast<__half2*>(shift_state)[static_cast<int64_t>(slot) * pairs + p] = norm2;
  }
}

}  // namespace

void cmix_tokenshift_forward_varlen_cuda(
    int batch_size,
    int total_tokens,
    int channels,
    int max_seqlen,
    torch::stable::Tensor x,
    torch::stable::Tensor shift_state,
    torch::stable::Tensor x_k,
    torch::stable::Tensor output,
    torch::stable::Tensor query_start_loc,
    torch::stable::Tensor state_indices,
    torch::stable::Tensor metadata_status,
    torch::stable::Tensor token_predecessor) {
  constexpr int threads = 256;
  constexpr int cmix_tokenshift_3d_b1_t_4096[] = {2, 4, 16, 64, 512};
  // Albatross uses grid=(channel_tiles,T,B).  This packed adaptation places
  // all token rows in grid.y, so actual total_tokens must satisfy the CUDA
  // y-dimension limit in addition to the upstream B/T tuned predicate.
  bool use_3d = channels == 4096 && batch_size <= 65535 &&
      max_seqlen <= 65535 && max_seqlen > 1 &&
      total_tokens <= kMaxGridDimYZ;
  if (use_3d && batch_size < 2) {
    use_3d = false;
    for (const int candidate : cmix_tokenshift_3d_b1_t_4096) {
      if (max_seqlen == candidate) {
        use_3d = true;
        break;
      }
    }
  }

  const auto stream = flashrwkv2::validation::current_cuda_stream();
  const int pairs = channels >> 1;
  if (use_3d) {
    // The upstream 3-D family is retained; y addresses packed tokens instead
    // of a rectangular T coordinate because varlen has no padding rows.
    cmix_tokenshift_varlen_kernel<true, false><<<
        dim3((pairs + threads - 1) / threads, total_tokens),
        dim3(threads),
        0,
        stream>>>(
        batch_size,
        total_tokens,
        channels,
        reinterpret_cast<const dtype*>(x.mutable_data_ptr()),
        reinterpret_cast<dtype*>(shift_state.mutable_data_ptr()),
        reinterpret_cast<const dtype*>(x_k.mutable_data_ptr()),
        reinterpret_cast<dtype*>(output.mutable_data_ptr()),
        token_predecessor.mutable_data_ptr<int>(),
        metadata_status.mutable_data_ptr<int>());
  } else {
    const int64_t total_pairs = static_cast<int64_t>(total_tokens) * pairs;
    if (max_seqlen == 1) {
      cmix_tokenshift_varlen_kernel<false, true><<<
          static_cast<int>((total_pairs + threads - 1) / threads),
          threads,
          0,
          stream>>>(
          batch_size,
          total_tokens,
          channels,
          reinterpret_cast<const dtype*>(x.mutable_data_ptr()),
          reinterpret_cast<dtype*>(shift_state.mutable_data_ptr()),
          reinterpret_cast<const dtype*>(x_k.mutable_data_ptr()),
          reinterpret_cast<dtype*>(output.mutable_data_ptr()),
          token_predecessor.mutable_data_ptr<int>(),
          metadata_status.mutable_data_ptr<int>());
    } else {
      cmix_tokenshift_varlen_kernel<false, false><<<
          static_cast<int>((total_pairs + threads - 1) / threads),
          threads,
          0,
          stream>>>(
          batch_size,
          total_tokens,
          channels,
          reinterpret_cast<const dtype*>(x.mutable_data_ptr()),
          reinterpret_cast<dtype*>(shift_state.mutable_data_ptr()),
          reinterpret_cast<const dtype*>(x_k.mutable_data_ptr()),
          reinterpret_cast<dtype*>(output.mutable_data_ptr()),
          token_predecessor.mutable_data_ptr<int>(),
          metadata_status.mutable_data_ptr<int>());
    }
  }

  if (max_seqlen > 1) {
    update_shift_state_last_varlen_kernel<<<batch_size, threads, 0, stream>>>(
        batch_size,
        channels,
        reinterpret_cast<const dtype*>(x.mutable_data_ptr()),
        reinterpret_cast<dtype*>(shift_state.mutable_data_ptr()),
        query_start_loc.mutable_data_ptr<int>(),
        state_indices.mutable_data_ptr<int>(),
        metadata_status.mutable_data_ptr<int>());
  }
  FLASHRWKV_CUDA_CHECK(cudaGetLastError());
}

std::vector<torch::stable::Tensor> cmix_res_ln_tokenshift_fused_forward_cuda(
    torch::stable::Tensor x,
    torch::stable::Tensor res,
    torch::stable::Tensor shift_state,
    torch::stable::Tensor weight,
    torch::stable::Tensor bias,
    torch::stable::Tensor x_k,
    torch::stable::Tensor token_predecessor,
    torch::stable::Tensor metadata_status,
    double eps) {
  auto res_out = torch::stable::empty_like(x);
  auto mixed = torch::stable::empty_like(x);
  // Match Albatross's C=4096 scalar-statistics launch.  Packed predecessor
  // addressing does not change the per-row channel parallelism.
  res_ln_cmix_tokenshift_fused_kernel<1024><<<
      static_cast<int>(x.size(0)), 1024, 0,
      flashrwkv2::validation::current_cuda_stream()>>>(
      x.mutable_data_ptr<dtype>(), res.mutable_data_ptr<dtype>(), shift_state.mutable_data_ptr<dtype>(),
      weight.mutable_data_ptr<dtype>(), bias.mutable_data_ptr<dtype>(), x_k.mutable_data_ptr<dtype>(),
      res_out.mutable_data_ptr<dtype>(), mixed.mutable_data_ptr<dtype>(),
      token_predecessor.mutable_data_ptr<int>(), metadata_status.mutable_data_ptr<int>(),
      x.size(0), static_cast<float>(eps));
  FLASHRWKV_CUDA_CHECK(cudaGetLastError());
  return {res_out, mixed};
}
