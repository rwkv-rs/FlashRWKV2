// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to BlinkDL/RWKV-LM
// Canonical source: RWKV-LM RWKV-v7/train_temp/cuda/rwkv7_cmix_bf16_v5.cu
// Source revision: 952102498e9ed367ea0a59ee64106916d474d30f.
// Local adaptation: propagate full ChannelMix nonzero initial and returned shift gradients.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <vector>

namespace {

__device__ inline __nv_bfloat162 load_bf16x2(const at::BFloat16* ptr) {
  return *reinterpret_cast<const __nv_bfloat162*>(ptr);
}
__device__ inline void store_bf16x2(at::BFloat16* ptr,
                                    __nv_bfloat162 value) {
  *reinterpret_cast<__nv_bfloat162*>(ptr) = value;
}
inline int64_t ceil_div(int64_t n, int64_t d) { return (n + d - 1) / d; }
constexpr int kThreads = 256;

__global__ void relu_square_backward_from_output_kernel(
    at::BFloat16* __restrict__ grad,
    const at::BFloat16* __restrict__ activation, int64_t pairs) {
  const int64_t pair =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pair >= pairs) return;
  const int64_t idx = pair * 2;
  const float2 g = __bfloat1622float2(load_bf16x2(grad + idx));
  const float2 out = __bfloat1622float2(load_bf16x2(activation + idx));
  store_bf16x2(
      grad + idx,
      __floats2bfloat162_rn(g.x * 2.0f * sqrtf(fmaxf(out.x, 0.0f)),
                            g.y * 2.0f * sqrtf(fmaxf(out.y, 0.0f))));
}

__global__ void statetune_cmix_tokenshift_backward_kernel(
    const at::BFloat16* __restrict__ grad_mixed,
    const at::BFloat16* __restrict__ grad_next,
    const at::BFloat16* __restrict__ x,
    const at::BFloat16* __restrict__ initial_shift,
    const at::BFloat16* __restrict__ x_k,
    at::BFloat16* __restrict__ grad_x,
    at::BFloat16* __restrict__ grad_initial,
    float* __restrict__ grad_x_k, int64_t b_size, int64_t t_size,
    int64_t c_size) {
  const int64_t bc_pair =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t pairs_per_row = c_size / 2;
  if (bc_pair >= b_size * pairs_per_row) return;
  const int64_t b = bc_pair / pairs_per_row;
  const int64_t c = (bc_pair % pairs_per_row) * 2;
  const int64_t base = b * t_size * c_size + c;
  const __nv_bfloat162 coefficient = load_bf16x2(x_k + c);
  const __nv_bfloat162 one_minus =
      __hsub2(__floats2bfloat162_rn(1.0f, 1.0f), coefficient);
  float2 coefficient_sum = make_float2(0.0f, 0.0f);

  for (int64_t t = 0; t < t_size; ++t) {
    const int64_t idx = base + t * c_size;
    __nv_bfloat162 dx =
        __hmul2(load_bf16x2(grad_mixed + idx), one_minus);
    if (t + 1 < t_size) {
      dx = __hadd2(dx, __hmul2(load_bf16x2(grad_mixed + idx + c_size),
                               coefficient));
    } else {
      dx = __hadd2(dx, load_bf16x2(grad_next + b * c_size + c));
    }
    store_bf16x2(grad_x + idx, dx);

    const __nv_bfloat162 previous =
        t == 0 ? load_bf16x2(initial_shift + b * c_size + c)
               : load_bf16x2(x + idx - c_size);
    const float2 delta =
        __bfloat1622float2(__hsub2(previous, load_bf16x2(x + idx)));
    const float2 grad = __bfloat1622float2(load_bf16x2(grad_mixed + idx));
    coefficient_sum.x += grad.x * delta.x;
    coefficient_sum.y += grad.y * delta.y;
  }
  store_bf16x2(grad_initial + b * c_size + c,
               __hmul2(load_bf16x2(grad_mixed + base), coefficient));
  atomicAdd(reinterpret_cast<float2*>(grad_x_k + c), coefficient_sum);
}

__global__ void cast_float_to_bf16_vec2_kernel(
    const float* __restrict__ source, at::BFloat16* __restrict__ target,
    int64_t pairs) {
  const int64_t pair =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pair >= pairs) return;
  const int64_t c = pair * 2;
  store_bf16x2(target + c,
               __floats2bfloat162_rn(source[c], source[c + 1]));
}

}  // namespace

std::vector<torch::Tensor> statetune_cmix_backward_cuda(
    torch::Tensor grad_output, torch::Tensor grad_next_shift,
    torch::Tensor x, torch::Tensor initial_shift, torch::Tensor x_k,
    torch::Tensor key_weight, torch::Tensor value_weight,
    torch::Tensor mixed, torch::Tensor activation) {
  const int64_t bt_size = x.size(0) * x.size(1);
  auto grad_output_2d =
      grad_output.view({bt_size, x.size(2)}).contiguous();
  auto mixed_2d = mixed.view({bt_size, x.size(2)});
  auto grad_value_weight =
      at::matmul(grad_output_2d.transpose(0, 1), activation);
  auto grad_hidden = at::matmul(grad_output_2d, value_weight).contiguous();
  auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t activation_pairs = activation.numel() / 2;
  relu_square_backward_from_output_kernel<<<
      static_cast<int>(ceil_div(activation_pairs, kThreads)), kThreads, 0,
      stream>>>(grad_hidden.data_ptr<at::BFloat16>(),
                activation.data_ptr<at::BFloat16>(), activation_pairs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto grad_key_weight =
      at::matmul(grad_hidden.transpose(0, 1), mixed_2d);
  auto grad_mixed_2d = at::matmul(grad_hidden, key_weight).contiguous();
  auto grad_mixed = grad_mixed_2d.view_as(x);

  auto grad_x = torch::empty_like(x);
  auto grad_initial = torch::empty_like(initial_shift);
  auto grad_x_k_fp32 =
      torch::zeros({x.size(2)}, x.options().dtype(torch::kFloat32));
  auto grad_x_k = torch::empty_like(x_k);
  const int64_t bc_pairs = x.size(0) * (x.size(2) / 2);
  statetune_cmix_tokenshift_backward_kernel<<<
      static_cast<int>(ceil_div(bc_pairs, kThreads)), kThreads, 0, stream>>>(
      grad_mixed.data_ptr<at::BFloat16>(),
      grad_next_shift.data_ptr<at::BFloat16>(), x.data_ptr<at::BFloat16>(),
      initial_shift.data_ptr<at::BFloat16>(), x_k.data_ptr<at::BFloat16>(),
      grad_x.data_ptr<at::BFloat16>(),
      grad_initial.data_ptr<at::BFloat16>(), grad_x_k_fp32.data_ptr<float>(),
      x.size(0), x.size(1), x.size(2));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const int64_t channel_pairs = x.size(2) / 2;
  cast_float_to_bf16_vec2_kernel<<<
      static_cast<int>(ceil_div(channel_pairs, kThreads)), kThreads, 0,
      stream>>>(grad_x_k_fp32.data_ptr<float>(),
                grad_x_k.data_ptr<at::BFloat16>(), channel_pairs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_x, grad_initial, grad_x_k, grad_key_weight.contiguous(),
          grad_value_weight.contiguous()};
}
