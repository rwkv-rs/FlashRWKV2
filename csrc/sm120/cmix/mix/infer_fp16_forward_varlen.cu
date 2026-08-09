// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the Albatross project
// SPDX-FileCopyrightText: Copyright contributors to the FlashRWKV2 project
//
// Source: BlinkDL/Albatross/faster3a_2607/cuda/rwkv7_fast_ops_fp16.cu,
// revision ee3308f6922e59f2166c7fac3c5a192340a2b48e.
//
// The CMix arithmetic and the two upstream launch families below are copied
// mechanically.  The only adaptation is the packed request address layer:
// cu_seqlens identifies the request boundary and state_indices identifies the
// request's shift-state slot.  No alternate CMix implementation is kept here.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cstdint>

at::Tensor linear_f16_lt_cfg_cuda(
    at::Tensor x,
    at::Tensor weight,
    int64_t workspace_mb,
    int64_t algo_index);
at::Tensor tmix_linear_forward_varlen_cuda(
    at::Tensor x,
    at::Tensor weight,
    bool weight_is_transposed,
    int64_t caller_group);

namespace {

constexpr unsigned int kMaxGridDimYZ = 65535;

using dtype = at::Half;

__device__ inline __half2 load_h2(const dtype* ptr) {
  return *reinterpret_cast<const __half2*>(ptr);
}

__device__ inline void store_h2(dtype* ptr, float x0, float x1) {
  *reinterpret_cast<__half2*>(ptr) = __floats2half2_rn(x0, x1);
}

__global__ void cmix_relu_square_varlen_kernel(
    const dtype* __restrict__ x,
    dtype* __restrict__ out,
    int64_t total_pairs) {
  const int64_t pair_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pair_idx >= total_pairs) {
    return;
  }
  const int64_t idx = pair_idx * 2;
  const float2 value = __half22float2(load_h2(x + idx));
  const float relu_x = fmaxf(value.x, 0.0f);
  const float relu_y = fmaxf(value.y, 0.0f);
  store_h2(out + idx, relu_x * relu_x, relu_y * relu_y);
}

__device__ inline int find_sequence(
    int token,
    int batch_size,
    const int* cu_seqlens) {
  int low = 0;
  int high = batch_size;
  while (low + 1 < high) {
    const int middle = (low + high) >> 1;
    if (cu_seqlens[middle] <= token) {
      low = middle;
    } else {
      high = middle;
    }
  }
  return low;
}

template <bool Grid3D, bool UpdateShift>
__global__ void cmix_mix_varlen_kernel(
    int batch_size,
    int total_tokens,
    int C,
    const dtype* __restrict__ x,
    dtype* __restrict__ shift_state,
    const dtype* __restrict__ x_k,
    dtype* __restrict__ out,
    const int* __restrict__ cu_seqlens,
    const int* __restrict__ state_indices,
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

  const int sequence = find_sequence(token, metadata_status[2], cu_seqlens);
  const int token_start = cu_seqlens[sequence];
  const int slot = state_indices[sequence];
  const int64_t previous_idx = token == token_start
      ? static_cast<int64_t>(slot) * C + c
      : static_cast<int64_t>(token - 1) * C + c;

  // This is the canonical Albatross cmix_mix arithmetic.
  const __half2 cur2 = load_h2(x + idx);
  const __half2 prev2 = token == token_start
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

}  // namespace

at::Tensor cmix_linear_ffn_down_forward_varlen_cuda(
    at::Tensor x, at::Tensor weight) {
  const int64_t rows = x.size(0);
  const bool canonical_4096_shape =
      x.size(1) == 16384 && weight.size(0) == 16384 &&
      weight.size(1) == 4096;
  if (canonical_4096_shape) {
    if (rows == 48) {
      // Exact FFN_DOWN_GEMM_4096[48] caller selection from Albatross.
      return linear_f16_lt_cfg_cuda(x, weight, 32, 1);
    }
    if (rows == 256) {
      // Exact FFN_DOWN_GEMM_4096[256] caller selection from Albatross.
      return linear_f16_lt_cfg_cuda(x, weight, 32, 5);
    }
  }
  // The unlisted path is the canonical runtime-layout linear caller.  It
  // selects the already migrated Albatross m=1 split-K or ordinary body; no
  // local GEMM replacement is introduced here.
  return tmix_linear_forward_varlen_cuda(x, weight, true, 0);
}

torch::Tensor cmix_relu_square_forward_varlen_cuda(torch::Tensor x) {
  auto out = torch::empty_like(x);
  constexpr int threads = 256;
  const int64_t total_pairs = x.numel() / 2;
  auto stream = at::cuda::getCurrentCUDAStream();
  cmix_relu_square_varlen_kernel<<<static_cast<int>(
      (total_pairs + threads - 1) / threads), threads, 0, stream>>>(
      reinterpret_cast<const dtype*>(x.data_ptr()),
      reinterpret_cast<dtype*>(out.data_ptr()),
      total_pairs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

void cmix_mix_forward_varlen_cuda(
    int batch_size,
    int total_tokens,
    int channels,
    int max_seqlen,
    torch::Tensor x,
    torch::Tensor shift_state,
    torch::Tensor x_k,
    torch::Tensor output,
    torch::Tensor query_start_loc,
    torch::Tensor state_indices,
    torch::Tensor metadata_status) {
  constexpr int threads = 256;
  constexpr int cmix_mix_3d_b1_t_4096[] = {2, 4, 16, 64, 512};
  // Albatross uses grid=(channel_tiles,T,B).  This packed adaptation places
  // all token rows in grid.y, so actual total_tokens must satisfy the CUDA
  // y-dimension limit in addition to the upstream B/T tuned predicate.
  bool use_3d = channels == 4096 && batch_size <= 65535 &&
      max_seqlen <= 65535 && max_seqlen > 1 &&
      total_tokens <= kMaxGridDimYZ;
  if (use_3d && batch_size < 2) {
    use_3d = false;
    for (const int candidate : cmix_mix_3d_b1_t_4096) {
      if (max_seqlen == candidate) {
        use_3d = true;
        break;
      }
    }
  }

  const auto stream = at::cuda::getCurrentCUDAStream();
  const int pairs = channels >> 1;
  if (use_3d) {
    // The upstream 3-D family is retained; y addresses packed tokens instead
    // of a rectangular T coordinate because varlen has no padding rows.
    cmix_mix_varlen_kernel<true, false><<<
        dim3((pairs + threads - 1) / threads, total_tokens),
        dim3(threads),
        0,
        stream>>>(
        batch_size,
        total_tokens,
        channels,
        reinterpret_cast<const dtype*>(x.data_ptr()),
        reinterpret_cast<dtype*>(shift_state.data_ptr()),
        reinterpret_cast<const dtype*>(x_k.data_ptr()),
        reinterpret_cast<dtype*>(output.data_ptr()),
        query_start_loc.data_ptr<int>(),
        state_indices.data_ptr<int>(),
        metadata_status.data_ptr<int>());
  } else {
    const int64_t total_pairs = static_cast<int64_t>(total_tokens) * pairs;
    if (max_seqlen == 1) {
      cmix_mix_varlen_kernel<false, true><<<
          static_cast<int>((total_pairs + threads - 1) / threads),
          threads,
          0,
          stream>>>(
          batch_size,
          total_tokens,
          channels,
          reinterpret_cast<const dtype*>(x.data_ptr()),
          reinterpret_cast<dtype*>(shift_state.data_ptr()),
          reinterpret_cast<const dtype*>(x_k.data_ptr()),
          reinterpret_cast<dtype*>(output.data_ptr()),
          query_start_loc.data_ptr<int>(),
          state_indices.data_ptr<int>(),
          metadata_status.data_ptr<int>());
    } else {
      cmix_mix_varlen_kernel<false, false><<<
          static_cast<int>((total_pairs + threads - 1) / threads),
          threads,
          0,
          stream>>>(
          batch_size,
          total_tokens,
          channels,
          reinterpret_cast<const dtype*>(x.data_ptr()),
          reinterpret_cast<dtype*>(shift_state.data_ptr()),
          reinterpret_cast<const dtype*>(x_k.data_ptr()),
          reinterpret_cast<dtype*>(output.data_ptr()),
          query_start_loc.data_ptr<int>(),
          state_indices.data_ptr<int>(),
          metadata_status.data_ptr<int>());
    }
  }

  if (max_seqlen > 1) {
    update_shift_state_last_varlen_kernel<<<batch_size, threads, 0, stream>>>(
        batch_size,
        channels,
        reinterpret_cast<const dtype*>(x.data_ptr()),
        reinterpret_cast<dtype*>(shift_state.data_ptr()),
        query_start_loc.data_ptr<int>(),
        state_indices.data_ptr<int>(),
        metadata_status.data_ptr<int>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
