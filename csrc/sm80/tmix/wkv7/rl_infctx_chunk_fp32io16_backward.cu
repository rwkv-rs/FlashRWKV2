// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright contributors to the FlashRWKV2 project
// Canonical module owner: tmix/wkv7; RL/Infctx is the workload.
// Source revision: FlashRWKV2 pre-refactor retained RL/Infctx local snapshot
// Original path: retained RL/Infctx materialized/recompute CUDA family
// Mechanically migrated from the retained RL/Infctx materialized affine
// implementation; raw decay-logit contract and module-local naming only.


#include "rl_infctx_chunk_fp32io16_replay.cuh"

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include "recurrent_decay.cuh"


namespace rl_infctx_replay_detail {

constexpr int kHeadSize = 64;

using flashrwkv2::wkv7::recurrent_retention;

template <typename io_t>
__device__ __forceinline__ float to_float(io_t value) {
  return static_cast<float>(value);
}

template <typename io_t>
__device__ __forceinline__ io_t from_float(float value) {
  return static_cast<io_t>(value);
}

struct OutputShared {
  float r[kHeadSize];
  float decay[kHeadSize];
  float k[kHeadSize];
  float a[kHeadSize];
  float b[kHeadSize];
};

template <typename io_t>
__global__ __launch_bounds__(kHeadSize, 2)
void emit_outputs_kernel(
    int num_heads,
    const int* __restrict__ chunk_token_starts,
    const int* __restrict__ chunk_token_ends,
    const float* __restrict__ boundary_ptr,
    const io_t* __restrict__ r_ptr,
    const io_t* __restrict__ decay_ptr,
    const io_t* __restrict__ decay_bias_ptr,
    const io_t* __restrict__ k_ptr,
    const io_t* __restrict__ v_ptr,
    const io_t* __restrict__ a_ptr,
    const io_t* __restrict__ b_ptr,
    io_t* __restrict__ output_ptr,
    float* __restrict__ state_dot_a_ptr,
    float scale) {
  const int linear_block = static_cast<int>(blockIdx.x);
  const int chunk_index = linear_block / num_heads;
  const int head_index = linear_block % num_heads;
  const int value_index = static_cast<int>(threadIdx.x);
  __shared__ OutputShared shared;

  const int64_t boundary_base =
      (static_cast<int64_t>(chunk_index) * num_heads + head_index) *
      kHeadSize * kHeadSize;
  float state[kHeadSize];
#pragma unroll
  for (int key_index = 0; key_index < kHeadSize; ++key_index) {
    state[key_index] =
        boundary_ptr[
            boundary_base + key_index * kHeadSize + value_index];
  }

  const int token_start = chunk_token_starts[chunk_index];
  const int token_end = chunk_token_ends[chunk_index];
  for (int token_index = token_start;
       token_index < token_end;
       ++token_index) {
    const int64_t input_index =
        (static_cast<int64_t>(token_index) * num_heads + head_index) *
            kHeadSize +
        value_index;
    shared.r[value_index] = to_float(r_ptr[input_index]);
    float decay_input = to_float(decay_ptr[input_index]);
    if (decay_bias_ptr != nullptr) {
      decay_input += to_float(
          decay_bias_ptr[head_index * kHeadSize + value_index]);
    }
    shared.decay[value_index] = recurrent_retention(decay_input);
    shared.k[value_index] = to_float(k_ptr[input_index]);
    shared.a[value_index] = to_float(a_ptr[input_index]);
    shared.b[value_index] = to_float(b_ptr[input_index]);
    const float value = to_float(v_ptr[input_index]);
    __syncthreads();

    float state_dot_a = 0.0f;
#pragma unroll
    for (int key_index = 0; key_index < kHeadSize; ++key_index) {
      state_dot_a =
          fmaf(shared.a[key_index], state[key_index], state_dot_a);
    }
    if (state_dot_a_ptr != nullptr) {
      state_dot_a_ptr[input_index] = state_dot_a;
    }

    float output = 0.0f;
#pragma unroll
    for (int key_index = 0; key_index < kHeadSize; ++key_index) {
      const float updated = fmaf(
          shared.k[key_index],
          value,
          fmaf(
              shared.b[key_index],
              state_dot_a,
              shared.decay[key_index] * state[key_index]));
      state[key_index] = updated;
      output = fmaf(shared.r[key_index], updated, output);
    }
    output_ptr[input_index] = from_float<io_t>(scale * output);
    __syncthreads();
  }
}

template <typename io_t>
void launch_replay(
    int num_chunks,
    int num_heads,
    const torch::Tensor& chunk_token_starts,
    const torch::Tensor& chunk_token_ends,
    const torch::Tensor& boundary,
    const torch::Tensor& r,
    const torch::Tensor& decay,
    const torch::Tensor& decay_bias,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor& output,
    torch::Tensor* state_dot_a,
    float scale,
    cudaStream_t stream) {
  emit_outputs_kernel<io_t>
      <<<num_chunks * num_heads, kHeadSize, 0, stream>>>(
          num_heads,
          chunk_token_starts.data_ptr<int>(),
          chunk_token_ends.data_ptr<int>(),
          boundary.data_ptr<float>(),
          r.data_ptr<io_t>(),
          decay.data_ptr<io_t>(),
          decay_bias.defined() ? decay_bias.data_ptr<io_t>() : nullptr,
          k.data_ptr<io_t>(),
          v.data_ptr<io_t>(),
          a.data_ptr<io_t>(),
          b.data_ptr<io_t>(),
          output.data_ptr<io_t>(),
          state_dot_a == nullptr
              ? nullptr
              : state_dot_a->data_ptr<float>(),
          scale);
}

}  // namespace rl_infctx_replay_detail

namespace rl_infctx_replay_tiled_detail {
template<typename io_t,int HeadSize>
__global__ void replay_columns(int num_heads,const int* starts,const int* ends,
 const float* boundary,const io_t* r,const io_t* decay,const io_t* decay_bias,
 const io_t* k,const io_t* v,const io_t* a,const io_t* b,io_t* output,
 float* state_dot_a,float scale) {
  const int value=blockIdx.x,head=blockIdx.y,chunk=blockIdx.z,lane=threadIdx.x;
  const int64_t bb=(static_cast<int64_t>(chunk)*num_heads+head)*HeadSize*HeadSize;
  float states[HeadSize/32];
#pragma unroll
  for(int q=0;q<HeadSize/32;++q){const int key=lane+q*32;states[q]=boundary[bb+static_cast<int64_t>(key)*HeadSize+value];}
  for(int token=starts[chunk];token<ends[chunk];++token){
    const int64_t tb=(static_cast<int64_t>(token)*num_heads+head)*HeadSize;
    float sa=0.0f;
    for(int q=0;q<HeadSize/32;++q){const int key=lane+q*32;sa=fmaf(static_cast<float>(a[tb+key]),states[q],sa);}
    for(int off=16;off;off>>=1)sa+=__shfl_down_sync(0xffffffffu,sa,off);
    sa=__shfl_sync(0xffffffffu,sa,0);
    if(lane==0)state_dot_a[tb+value]=sa;
    float y=0.0f;const float vv=static_cast<float>(v[tb+value]);
    for(int q=0;q<HeadSize/32;++q){
      const int key=lane+q*32;
      float dl=static_cast<float>(decay[tb+key]);
      if(decay_bias)dl+=static_cast<float>(decay_bias[head*HeadSize+key]);
      states[q]=flashrwkv2::wkv7::recurrent_retention(dl)*states[q]+
        static_cast<float>(b[tb+key])*sa+static_cast<float>(k[tb+key])*vv;
      y=fmaf(static_cast<float>(r[tb+key]),states[q],y);
    }
    for(int off=16;off;off>>=1)y+=__shfl_down_sync(0xffffffffu,y,off);
    if(lane==0)output[tb+value]=static_cast<io_t>(scale*y);
  }
}
}


void launch_rl_infctx_chunk_replay_fp32_from_decay_logits(
    int num_chunks,
    int num_heads,
    const torch::Tensor& chunk_token_starts,
    const torch::Tensor& chunk_token_ends,
    const torch::Tensor& boundary,
    const torch::Tensor& r,
    const torch::Tensor& decay_logits,
    const torch::Tensor& decay_bias,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor& output,
    torch::Tensor* state_dot_a,
    float scale,
    cudaStream_t stream) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      r.scalar_type(),
      "flashrwkv2_rl_infctx_chunk_replay_fp32",
      [&] {
        rl_infctx_replay_detail::launch_replay<scalar_t>(
            num_chunks, num_heads, chunk_token_starts, chunk_token_ends,
            boundary, r, decay_logits, decay_bias, k, v, a, b, output,
            state_dot_a, scale, stream);
      });
}




void rl_infctx_tmix_wkv7_chunk_fp32io16_backward_replay_cuda(
    torch::Tensor chunk_token_starts,
    torch::Tensor chunk_token_ends,
    torch::Tensor boundary,
    torch::Tensor r,
    torch::Tensor decay_logits,
    torch::Tensor decay_bias,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor output,
    torch::Tensor state_dot_a,
    double scale) {
  const c10::cuda::CUDAGuard device_guard(boundary.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int num_chunks = static_cast<int>(chunk_token_starts.numel());
  const int num_heads = static_cast<int>(r.size(1));
  const int head_size = static_cast<int>(r.size(2));
  if (head_size == 64) {
    launch_rl_infctx_chunk_replay_fp32_from_decay_logits(
      num_chunks, num_heads, chunk_token_starts, chunk_token_ends, boundary,
      r, decay_logits, decay_bias, k, v, a, b, output, &state_dot_a,
      static_cast<float>(scale), stream);
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, r.scalar_type(),
        "flashrwkv2_rl_replay_tiled", [&] {
          const dim3 grid(head_size, num_heads, num_chunks);
          if (head_size == 128)
            rl_infctx_replay_tiled_detail::replay_columns<scalar_t,128><<<grid,32,0,stream>>>(num_heads,chunk_token_starts.data_ptr<int>(),chunk_token_ends.data_ptr<int>(),boundary.data_ptr<float>(),r.data_ptr<scalar_t>(),decay_logits.data_ptr<scalar_t>(),decay_bias.defined()?decay_bias.data_ptr<scalar_t>():nullptr,k.data_ptr<scalar_t>(),v.data_ptr<scalar_t>(),a.data_ptr<scalar_t>(),b.data_ptr<scalar_t>(),output.data_ptr<scalar_t>(),state_dot_a.data_ptr<float>(),static_cast<float>(scale));
          else
            rl_infctx_replay_tiled_detail::replay_columns<scalar_t,256><<<grid,32,0,stream>>>(num_heads,chunk_token_starts.data_ptr<int>(),chunk_token_ends.data_ptr<int>(),boundary.data_ptr<float>(),r.data_ptr<scalar_t>(),decay_logits.data_ptr<scalar_t>(),decay_bias.defined()?decay_bias.data_ptr<scalar_t>():nullptr,k.data_ptr<scalar_t>(),v.data_ptr<scalar_t>(),a.data_ptr<scalar_t>(),b.data_ptr<scalar_t>(),output.data_ptr<scalar_t>(),state_dot_a.data_ptr<float>(),static_cast<float>(scale));
        });
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
