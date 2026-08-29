// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to BlinkDL/Albatross
// Native-private VRes implementation owned by TMix WKV Prepare.
// Albatross source revision: ee3308f6922e59f2166c7fac3c5a192340a2b48e.

#include "validation.h"

#include <cuda_fp16.h>

namespace {

using dtype = torch::headeronly::Half;

inline int64_t ceil_div(int64_t n, int64_t d) {
  return (n + d - 1) / d;
}

__device__ inline float sigmoid_fast(float x) {
  return 1.0f / (1.0f + __expf(-x));
}

__global__ void wkv_prepare_vres_kernel(
    int channels,
    int64_t elements,
    const dtype* __restrict__ value,
    const dtype* __restrict__ v_first,
    const dtype* __restrict__ v0,
    const dtype* __restrict__ delta,
    dtype* __restrict__ output) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) {
    return;
  }
  const int channel = static_cast<int>(index % channels);
  const float current = __half2float(*reinterpret_cast<const __half*>(value + index));
  const float first = __half2float(*reinterpret_cast<const __half*>(v_first + index));
  const float logits =
      __half2float(*reinterpret_cast<const __half*>(v0 + channel)) +
      __half2float(*reinterpret_cast<const __half*>(delta + index));
  const float gate = sigmoid_fast(logits);
  *reinterpret_cast<__half*>(output + index) =
      __float2half_rn(fmaf(first - current, gate, current));
}

template <int Threads>
__global__ __launch_bounds__(Threads) void wkv_prepare_vres_vec2_kernel(
    int channels,
    const dtype* __restrict__ value,
    const dtype* __restrict__ v_first,
    const dtype* __restrict__ v0,
    const dtype* __restrict__ delta,
    dtype* __restrict__ output,
    int64_t rows) {
  const int pair = static_cast<int>(blockIdx.x) * Threads + threadIdx.x;
  const int pairs_per_row = channels >> 1;
  const int64_t row = blockIdx.y;
  if (pair >= pairs_per_row || row >= rows) {
    return;
  }
  const int channel = pair << 1;
  const int64_t index = row * channels + channel;
  const float2 current = __half22float2(
      *reinterpret_cast<const __half2*>(value + index));
  const float2 first = __half22float2(
      *reinterpret_cast<const __half2*>(v_first + index));
  const float2 base = __half22float2(
      *reinterpret_cast<const __half2*>(v0 + channel));
  const float2 change = __half22float2(
      *reinterpret_cast<const __half2*>(delta + index));
  const float gate0 = sigmoid_fast(base.x + change.x);
  const float gate1 = sigmoid_fast(base.y + change.y);
  *reinterpret_cast<__half2*>(output + index) = __floats2half2_rn(
      fmaf(first.x - current.x, gate0, current.x),
      fmaf(first.y - current.y, gate1, current.y));
}

template <int Threads>
void launch_vec2(
    int channels,
    int64_t rows,
    const torch::stable::Tensor& value,
    const torch::stable::Tensor& v_first,
    const torch::stable::Tensor& v0,
    const torch::stable::Tensor& delta,
    torch::stable::Tensor& output,
    cudaStream_t stream) {
  const int pairs_per_row = channels >> 1;
  wkv_prepare_vres_vec2_kernel<Threads><<<
      dim3(static_cast<unsigned int>(ceil_div(pairs_per_row, Threads)),
           static_cast<unsigned int>(rows), 1),
      Threads, 0, stream>>>(
      channels, value.mutable_data_ptr<dtype>(), v_first.mutable_data_ptr<dtype>(),
      v0.mutable_data_ptr<dtype>(), delta.mutable_data_ptr<dtype>(), output.mutable_data_ptr<dtype>(),
      rows);
}

}  // namespace

void wkv_prepare_vres_forward_varlen_cuda(
    int total_tokens,
    int channels,
    torch::stable::Tensor value,
    torch::stable::Tensor v_first,
    torch::stable::Tensor v0,
    torch::stable::Tensor delta,
    torch::stable::Tensor output) {
  auto stream = flashrwkv2::validation::current_cuda_stream();
  const int64_t rows = total_tokens;
  const bool use_vec2 = channels == 4096 && rows >= 64 && rows <= 65535;
  if (use_vec2) {
    if (rows < 256) {
      launch_vec2<128>(
          channels, rows, value, v_first, v0, delta, output, stream);
    } else {
      launch_vec2<256>(
          channels, rows, value, v_first, v0, delta, output, stream);
    }
  } else {
    constexpr int threads = 256;
    const int64_t elements = static_cast<int64_t>(total_tokens) * channels;
    wkv_prepare_vres_kernel<<<
        static_cast<unsigned int>(ceil_div(elements, threads)),
        threads, 0, stream>>>(
        channels, elements, value.mutable_data_ptr<dtype>(), v_first.mutable_data_ptr<dtype>(),
        v0.mutable_data_ptr<dtype>(), delta.mutable_data_ptr<dtype>(), output.mutable_data_ptr<dtype>());
  }
  FLASHRWKV_CUDA_CHECK(cudaGetLastError());
}
