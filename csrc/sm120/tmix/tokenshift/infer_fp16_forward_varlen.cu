// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the Albatross project
//
// Source: BlinkDL/Albatross/faster3a_2607/cuda/rwkv7_fast_ops_fp16.cu,
// revision ee3308f6922e59f2166c7fac3c5a192340a2b48e.
//
// The tokenshift arithmetic and its two upstream launch families are copied from
// that file.  The only local adaptation is the packed varlen boundary:
// cu_seqlens selects the previous token and state_indices selects the state
// slot.  The final-token state closure is the upstream update-shift stage with
// those two address calculations changed for packed storage.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cstdint>
#include <vector>

using dtype = at::Half;

namespace {

constexpr unsigned int kMaxGridDimYZ = 65535;

inline int64_t ceil_div(int64_t n, int64_t d) {
  return (n + d - 1) / d;
}

__device__ inline __half2 load_h2(const dtype* ptr) {
  return *reinterpret_cast<const __half2*>(ptr);
}

__device__ inline void store_h2(dtype* ptr, float x0, float x1) {
  *reinterpret_cast<__half2*>(ptr) = __floats2half2_rn(x0, x1);
}

__device__ __forceinline__ int find_sequence(
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

template <typename T>
__device__ __forceinline__ void fill_invalid(
    int64_t block_index,
    int64_t block_count,
    int64_t elements,
    T* output) {
  const T value = __float2half(__int_as_float(0x7fffffff));
  for (int64_t i = block_index * blockDim.x + threadIdx.x; i < elements;
       i += block_count * blockDim.x) {
    output[i] = value;
  }
}

// Upstream tmix_tokenshift_kernel, with only the packed token/state address
// calculation added.  Grid3D preserves the Albatross 2-D CTA family for the
// selected C==4096 shapes; it does not introduce a new arithmetic path.
template <bool Grid3D, bool UpdateShift>
__global__ void tmix_tokenshift_kernel(
    int batch_size,
    int total_tokens,
    int C,
    const dtype* __restrict__ x,
    dtype* __restrict__ shift_state,
    const dtype* __restrict__ x_r,
    const dtype* __restrict__ x_w,
    const dtype* __restrict__ x_k,
    const dtype* __restrict__ x_v,
    const dtype* __restrict__ x_a,
    const dtype* __restrict__ x_g,
    dtype* __restrict__ out_r,
    dtype* __restrict__ out_w,
    dtype* __restrict__ out_k,
    dtype* __restrict__ out_v,
    dtype* __restrict__ out_a,
    dtype* __restrict__ out_g,
    const int* __restrict__ cu_seqlens,
    const int* __restrict__ state_indices,
    const int* __restrict__ metadata_status) {
  const int c_pairs = C >> 1;
  const int64_t pair_index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int token = Grid3D
      ? static_cast<int>(blockIdx.y)
      : static_cast<int>(pair_index / c_pairs);
  const int pair = Grid3D
      ? static_cast<int>(pair_index)
      : static_cast<int>(pair_index % c_pairs);
  if (pair >= c_pairs || token >= total_tokens) {
    return;
  }

  const int64_t block_index = Grid3D
      ? static_cast<int64_t>(blockIdx.y) * gridDim.x + blockIdx.x
      : static_cast<int64_t>(blockIdx.x);
  const int64_t block_count = Grid3D
      ? static_cast<int64_t>(gridDim.x) * gridDim.y
      : static_cast<int64_t>(gridDim.x);
  if (metadata_status[0] != 0) {
    const int64_t elements = static_cast<int64_t>(total_tokens) * C;
    fill_invalid(block_index, block_count, elements, out_r);
    fill_invalid(block_index, block_count, elements, out_w);
    fill_invalid(block_index, block_count, elements, out_k);
    fill_invalid(block_index, block_count, elements, out_v);
    fill_invalid(block_index, block_count, elements, out_a);
    fill_invalid(block_index, block_count, elements, out_g);
    return;
  }
  if (token >= metadata_status[1]) {
    return;
  }

  const int sequence = find_sequence(token, metadata_status[2], cu_seqlens);
  const int token_start = cu_seqlens[sequence];
  const int slot = state_indices[sequence];
  const int c = pair << 1;
  const int64_t idx = static_cast<int64_t>(token) * C + c;
  const int64_t previous_idx = token == token_start
      ? static_cast<int64_t>(slot) * C + c
      : idx - C;

  const __half2 cur2 = load_h2(x + idx);
  const __half2 prev2 = token == token_start
      ? load_h2(shift_state + previous_idx)
      : load_h2(x + previous_idx);
  const float2 cur = __half22float2(cur2);
  const float2 prev = __half22float2(prev2);
  const float dx0 = prev.x - cur.x;
  const float dx1 = prev.y - cur.y;

  const float2 xr = __half22float2(load_h2(x_r + c));
  const float2 xw = __half22float2(load_h2(x_w + c));
  const float2 xk = __half22float2(load_h2(x_k + c));
  const float2 xv = __half22float2(load_h2(x_v + c));
  const float2 xa = __half22float2(load_h2(x_a + c));
  const float2 xg = __half22float2(load_h2(x_g + c));

  store_h2(out_r + idx, cur.x + dx0 * xr.x, cur.y + dx1 * xr.y);
  store_h2(out_w + idx, cur.x + dx0 * xw.x, cur.y + dx1 * xw.y);
  store_h2(out_k + idx, cur.x + dx0 * xk.x, cur.y + dx1 * xk.y);
  store_h2(out_v + idx, cur.x + dx0 * xv.x, cur.y + dx1 * xv.y);
  store_h2(out_a + idx, cur.x + dx0 * xa.x, cur.y + dx1 * xa.y);
  store_h2(out_g + idx, cur.x + dx0 * xg.x, cur.y + dx1 * xg.y);

  if constexpr (UpdateShift) {
    // This specialization is used only when every request has T==1, which
    // is the same state-update boundary as upstream tmix_tokenshift.
    *reinterpret_cast<__half2*>(
        shift_state + static_cast<int64_t>(slot) * C + c) = cur2;
  }
}

// Upstream update_shift_state_last_kernel, adapted from [B,T,C] to packed
// rows and scheduler-owned state slots.
__global__ void update_shift_state_last_kernel(
    int batch_size,
    int channels,
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
  for (int channel = threadIdx.x; channel < channels; channel += blockDim.x) {
    shift_state[static_cast<int64_t>(slot) * channels + channel] =
        x[static_cast<int64_t>(token) * channels + channel];
  }
}

bool use_tmix_tokenshift_grid3d(
    int batch_size,
    int total_tokens,
    int max_seqlen,
    int channels) {
  // Albatross launches this family as grid=(channel_tiles,T,B).  Packed
  // varlen storage flattens T/B into grid.y=total_tokens, so the upstream
  // scheduler predicate and the actual CUDA launch extent are independent
  // requirements.
  constexpr int kB1T4096[] = {2, 4, 16, 64, 512};
  if (max_seqlen == 1 || channels != 4096 || batch_size > 65535 ||
      max_seqlen > 65535 || total_tokens > kMaxGridDimYZ) {
    return false;
  }
  if (batch_size >= 2) {
    return true;
  }
  for (const int value : kB1T4096) {
    if (max_seqlen == value) {
      return true;
    }
  }
  return false;
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
__global__ __launch_bounds__(Threads, 1) void res_ln_tmix_tokenshift_fused_kernel(
    const dtype* __restrict__ x,
    const dtype* __restrict__ res,
    dtype* __restrict__ shift_state,
    const dtype* __restrict__ weight,
    const dtype* __restrict__ bias,
    const dtype* __restrict__ x_r,
    const dtype* __restrict__ x_w,
    const dtype* __restrict__ x_k,
    const dtype* __restrict__ x_v,
    const dtype* __restrict__ x_a,
    const dtype* __restrict__ x_g,
    dtype* __restrict__ res_out,
    dtype* __restrict__ out_r,
    dtype* __restrict__ out_w,
    dtype* __restrict__ out_k,
    dtype* __restrict__ out_v,
    dtype* __restrict__ out_a,
    dtype* __restrict__ out_g,
    const int* __restrict__ state_indices,
    const int* __restrict__ metadata_status,
    float eps) {
  constexpr int C = 4096;
  constexpr int pairs = C / 2;
  if (metadata_status[0] != 0) {
    const __half2 invalid = __floats2half2_rn(
        __int_as_float(0x7fffffff), __int_as_float(0x7fffffff));
    for (int p = threadIdx.x; p < pairs; p += Threads) {
      reinterpret_cast<__half2*>(res_out)[p] = invalid;
      reinterpret_cast<__half2*>(out_r)[p] = invalid;
      reinterpret_cast<__half2*>(out_w)[p] = invalid;
      reinterpret_cast<__half2*>(out_k)[p] = invalid;
      reinterpret_cast<__half2*>(out_v)[p] = invalid;
      reinterpret_cast<__half2*>(out_a)[p] = invalid;
      reinterpret_cast<__half2*>(out_g)[p] = invalid;
    }
    return;
  }
  // CUDA Graph replay may reduce the active batch to zero while retaining
  // the captured physical buffers.  Do not inspect state_indices in that case.
  if (metadata_status[1] == 0 || metadata_status[2] == 0) return;
  const int slot = state_indices[0];
  float sum = 0.0f;
#pragma unroll
  for (int p = threadIdx.x; p < pairs; p += Threads) {
    const float2 xv = __half22float2(reinterpret_cast<const __half2*>(x)[p]);
    const float2 rv = __half22float2(reinterpret_cast<const __half2*>(res)[p]);
    sum += xv.x + rv.x + xv.y + rv.y;
  }
  const float mean = block_sum<Threads>(sum) / C;
  float var = 0.0f;
#pragma unroll
  for (int p = threadIdx.x; p < pairs; p += Threads) {
    const float2 xv = __half22float2(reinterpret_cast<const __half2*>(x)[p]);
    const float2 rv = __half22float2(reinterpret_cast<const __half2*>(res)[p]);
    const float d0 = xv.x + rv.x - mean;
    const float d1 = xv.y + rv.y - mean;
    var += d0 * d0 + d1 * d1;
  }
  const float rstd = rsqrtf(block_sum<Threads>(var) / C + eps);
#pragma unroll
  for (int p = threadIdx.x; p < pairs; p += Threads) {
    const float2 xv = __half22float2(reinterpret_cast<const __half2*>(x)[p]);
    const float2 rv = __half22float2(reinterpret_cast<const __half2*>(res)[p]);
    const float2 w = __half22float2(reinterpret_cast<const __half2*>(weight)[p]);
    const float2 b = __half22float2(reinterpret_cast<const __half2*>(bias)[p]);
    const float2 prev = __half22float2(
        reinterpret_cast<const __half2*>(shift_state)[static_cast<int64_t>(slot) * pairs + p]);
    const float s0 = xv.x + rv.x;
    const float s1 = xv.y + rv.y;
    const __half2 norm2 = __floats2half2_rn(
        (s0 - mean) * rstd * w.x + b.x,
        (s1 - mean) * rstd * w.y + b.y);
    const float2 norm = __half22float2(norm2);
    const float d0 = prev.x - norm.x;
    const float d1 = prev.y - norm.y;
#define STORE_MIX(dst, coeff) do { \
      const float2 m = __half22float2(reinterpret_cast<const __half2*>(coeff)[p]); \
      reinterpret_cast<__half2*>(dst)[p] = __floats2half2_rn( \
          norm.x + d0 * m.x, norm.y + d1 * m.y); \
    } while (0)
    reinterpret_cast<__half2*>(res_out)[p] = __floats2half2_rn(s0, s1);
    STORE_MIX(out_r, x_r);
    STORE_MIX(out_w, x_w);
    STORE_MIX(out_k, x_k);
    STORE_MIX(out_v, x_v);
    STORE_MIX(out_a, x_a);
    STORE_MIX(out_g, x_g);
#undef STORE_MIX
    reinterpret_cast<__half2*>(shift_state)[static_cast<int64_t>(slot) * pairs + p] = norm2;
  }
}

}  // namespace

void tmix_tokenshift_forward_varlen(
    int batch_size,
    int total_tokens,
    int channels,
    int max_seqlen,
    torch::Tensor x,
    torch::Tensor shift_state,
    torch::Tensor x_r,
    torch::Tensor x_w,
    torch::Tensor x_k,
    torch::Tensor x_v,
    torch::Tensor x_a,
    torch::Tensor x_g,
    torch::Tensor query_start_loc,
    torch::Tensor state_indices,
    torch::Tensor metadata_status,
    std::vector<torch::Tensor>& outputs) {
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int pairs = channels / 2;
  const bool use_grid3d =
      use_tmix_tokenshift_grid3d(
          batch_size, total_tokens, max_seqlen, channels);
  const bool update_in_mix = max_seqlen == 1;

  if (use_grid3d) {
    const dim3 grid(
        static_cast<unsigned int>(ceil_div(pairs, 256)),
        static_cast<unsigned int>(total_tokens));
    tmix_tokenshift_kernel<true, false><<<grid, 256, 0, stream>>>(
        batch_size, total_tokens, channels,
        x.data_ptr<dtype>(), shift_state.data_ptr<dtype>(),
        x_r.data_ptr<dtype>(), x_w.data_ptr<dtype>(), x_k.data_ptr<dtype>(),
        x_v.data_ptr<dtype>(), x_a.data_ptr<dtype>(), x_g.data_ptr<dtype>(),
        outputs[0].data_ptr<dtype>(), outputs[1].data_ptr<dtype>(),
        outputs[2].data_ptr<dtype>(), outputs[3].data_ptr<dtype>(),
        outputs[4].data_ptr<dtype>(), outputs[5].data_ptr<dtype>(),
        query_start_loc.data_ptr<int>(), state_indices.data_ptr<int>(),
        metadata_status.data_ptr<int>());
  } else {
    const dim3 grid(static_cast<unsigned int>(ceil_div(
        static_cast<int64_t>(total_tokens) * pairs, 256)));
    if (update_in_mix) {
      tmix_tokenshift_kernel<false, true><<<grid, 256, 0, stream>>>(
          batch_size, total_tokens, channels,
          x.data_ptr<dtype>(), shift_state.data_ptr<dtype>(),
          x_r.data_ptr<dtype>(), x_w.data_ptr<dtype>(), x_k.data_ptr<dtype>(),
          x_v.data_ptr<dtype>(), x_a.data_ptr<dtype>(), x_g.data_ptr<dtype>(),
          outputs[0].data_ptr<dtype>(), outputs[1].data_ptr<dtype>(),
          outputs[2].data_ptr<dtype>(), outputs[3].data_ptr<dtype>(),
          outputs[4].data_ptr<dtype>(), outputs[5].data_ptr<dtype>(),
          query_start_loc.data_ptr<int>(), state_indices.data_ptr<int>(),
          metadata_status.data_ptr<int>());
    } else {
      tmix_tokenshift_kernel<false, false><<<grid, 256, 0, stream>>>(
          batch_size, total_tokens, channels,
          x.data_ptr<dtype>(), shift_state.data_ptr<dtype>(),
          x_r.data_ptr<dtype>(), x_w.data_ptr<dtype>(), x_k.data_ptr<dtype>(),
          x_v.data_ptr<dtype>(), x_a.data_ptr<dtype>(), x_g.data_ptr<dtype>(),
          outputs[0].data_ptr<dtype>(), outputs[1].data_ptr<dtype>(),
          outputs[2].data_ptr<dtype>(), outputs[3].data_ptr<dtype>(),
          outputs[4].data_ptr<dtype>(), outputs[5].data_ptr<dtype>(),
          query_start_loc.data_ptr<int>(), state_indices.data_ptr<int>(),
          metadata_status.data_ptr<int>());
    }
  }

  if (!update_in_mix) {
    update_shift_state_last_kernel<<<batch_size, 256, 0, stream>>>(
        batch_size, channels, x.data_ptr<dtype>(), shift_state.data_ptr<dtype>(),
        query_start_loc.data_ptr<int>(), state_indices.data_ptr<int>(),
        metadata_status.data_ptr<int>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<torch::Tensor> tmix_res_ln_tokenshift_fused_forward_cuda(
    torch::Tensor x,
    torch::Tensor res,
    torch::Tensor shift_state,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor x_r,
    torch::Tensor x_w,
    torch::Tensor x_k,
    torch::Tensor x_v,
    torch::Tensor x_a,
    torch::Tensor x_g,
    torch::Tensor state_indices,
    torch::Tensor metadata_status,
    double eps) {
  std::vector<torch::Tensor> outputs;
  outputs.reserve(7);
  for (int i = 0; i < 7; ++i) outputs.push_back(torch::empty_like(x));
  res_ln_tmix_tokenshift_fused_kernel<256><<<
      1, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      x.data_ptr<dtype>(), res.data_ptr<dtype>(), shift_state.data_ptr<dtype>(),
      weight.data_ptr<dtype>(), bias.data_ptr<dtype>(), x_r.data_ptr<dtype>(),
      x_w.data_ptr<dtype>(), x_k.data_ptr<dtype>(), x_v.data_ptr<dtype>(),
      x_a.data_ptr<dtype>(), x_g.data_ptr<dtype>(), outputs[0].data_ptr<dtype>(),
      outputs[1].data_ptr<dtype>(), outputs[2].data_ptr<dtype>(),
      outputs[3].data_ptr<dtype>(), outputs[4].data_ptr<dtype>(),
      outputs[5].data_ptr<dtype>(), outputs[6].data_ptr<dtype>(),
      state_indices.data_ptr<int>(), metadata_status.data_ptr<int>(),
      static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return outputs;
}
