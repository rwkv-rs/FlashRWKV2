// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to BlinkDL/RWKV-LM
// Canonical source: RWKV-LM RWKV-v7/train_temp/cuda/rwkv7_tmix_tokenshift_bf16_v5.cu
// Source revision: 952102498e9ed367ea0a59ee64106916d474d30f.
// Local adaptation: use the caller's initial shift and return the last input.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <vector>

namespace {

std::vector<torch::Tensor> empty_like_pack(
    const torch::Tensor& reference, int64_t count) {
  auto storage = torch::empty({count * reference.numel()}, reference.options());
  std::vector<torch::Tensor> outputs;
  outputs.reserve(count);
  for (int64_t index = 0; index < count; ++index) {
    outputs.push_back(storage.as_strided(
        reference.sizes(), reference.strides(), index * reference.numel()));
  }
  return outputs;
}

__device__ inline __nv_bfloat162 load_bf16x2(const at::BFloat16* ptr) {
  return *reinterpret_cast<const __nv_bfloat162*>(ptr);
}

__device__ inline void store_bf16x2(at::BFloat16* ptr,
                                    __nv_bfloat162 value) {
  *reinterpret_cast<__nv_bfloat162*>(ptr) = value;
}

inline int64_t ceil_div(int64_t n, int64_t d) { return (n + d - 1) / d; }

__global__ void statetune_tmix_tokenshift_forward_kernel(
    const at::BFloat16* __restrict__ x,
    const at::BFloat16* __restrict__ initial_shift,
    const at::BFloat16* __restrict__ x_r,
    const at::BFloat16* __restrict__ x_w,
    const at::BFloat16* __restrict__ x_k,
    const at::BFloat16* __restrict__ x_v,
    const at::BFloat16* __restrict__ x_a,
    const at::BFloat16* __restrict__ x_g,
    at::BFloat16* __restrict__ out_r,
    at::BFloat16* __restrict__ out_w,
    at::BFloat16* __restrict__ out_k,
    at::BFloat16* __restrict__ out_v,
    at::BFloat16* __restrict__ out_a,
    at::BFloat16* __restrict__ out_g, int64_t bt_size, int64_t t_size,
    int64_t c_size) {
  const int64_t pair_idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t pairs_per_row = c_size / 2;
  if (pair_idx >= bt_size * pairs_per_row) return;

  const int64_t bt = pair_idx / pairs_per_row;
  const int64_t c = (pair_idx % pairs_per_row) * 2;
  const int64_t idx = bt * c_size + c;
  const int64_t t = bt % t_size;
  const int64_t b = bt / t_size;

  const __nv_bfloat162 current = load_bf16x2(x + idx);
  const __nv_bfloat162 previous =
      t == 0 ? load_bf16x2(initial_shift + b * c_size + c)
             : load_bf16x2(x + idx - c_size);
  const __nv_bfloat162 delta = __hsub2(previous, current);

  store_bf16x2(out_r + idx,
               __hadd2(current, __hmul2(delta, load_bf16x2(x_r + c))));
  store_bf16x2(out_w + idx,
               __hadd2(current, __hmul2(delta, load_bf16x2(x_w + c))));
  store_bf16x2(out_k + idx,
               __hadd2(current, __hmul2(delta, load_bf16x2(x_k + c))));
  store_bf16x2(out_v + idx,
               __hadd2(current, __hmul2(delta, load_bf16x2(x_v + c))));
  store_bf16x2(out_a + idx,
               __hadd2(current, __hmul2(delta, load_bf16x2(x_a + c))));
  store_bf16x2(out_g + idx,
               __hadd2(current, __hmul2(delta, load_bf16x2(x_g + c))));
}

}  // namespace

std::vector<torch::Tensor> statetune_tmix_tokenshift_forward_cuda(
    torch::Tensor x, torch::Tensor initial_shift, torch::Tensor x_r,
    torch::Tensor x_w, torch::Tensor x_k, torch::Tensor x_v,
    torch::Tensor x_a, torch::Tensor x_g) {
  auto outputs = empty_like_pack(x, 6);
  auto& out_r = outputs[0];
  auto& out_w = outputs[1];
  auto& out_k = outputs[2];
  auto& out_v = outputs[3];
  auto& out_a = outputs[4];
  auto& out_g = outputs[5];
  const int64_t bt_size = x.size(0) * x.size(1);
  const int64_t total_pairs = bt_size * (x.size(2) / 2);
  constexpr int threads = 256;
  const int blocks = static_cast<int>(ceil_div(total_pairs, threads));
  auto stream = at::cuda::getCurrentCUDAStream();
  statetune_tmix_tokenshift_forward_kernel<<<blocks, threads, 0, stream>>>(
      x.data_ptr<at::BFloat16>(), initial_shift.data_ptr<at::BFloat16>(),
      x_r.data_ptr<at::BFloat16>(), x_w.data_ptr<at::BFloat16>(),
      x_k.data_ptr<at::BFloat16>(), x_v.data_ptr<at::BFloat16>(),
      x_a.data_ptr<at::BFloat16>(), x_g.data_ptr<at::BFloat16>(),
      out_r.data_ptr<at::BFloat16>(), out_w.data_ptr<at::BFloat16>(),
      out_k.data_ptr<at::BFloat16>(), out_v.data_ptr<at::BFloat16>(),
      out_a.data_ptr<at::BFloat16>(), out_g.data_ptr<at::BFloat16>(), bt_size,
      x.size(1), x.size(2));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto next_shift = x.select(1, x.size(1) - 1).contiguous();
  outputs.push_back(next_shift);
  return outputs;
}
