// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the FlashRWKV2 project

#pragma once

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Macros.h>
#include <torch/headeronly/util/BFloat16.h>
#include <torch/headeronly/util/Half.h>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <optional>
#include <tuple>
#include <utility>
#include <vector>

namespace flashrwkv2::validation {

inline void check_cuda(
    cudaError_t error,
    const char* expression,
    const char* file,
    int line) {
  STD_TORCH_CHECK(
      error == cudaSuccess,
      expression,
      " failed at ",
      file,
      ':',
      line,
      ": ",
      cudaGetErrorString(error));
}

inline cudaStream_t current_cuda_stream() {
  int device_index = -1;
  check_cuda(cudaGetDevice(&device_index), "cudaGetDevice", __FILE__, __LINE__);
  void* stream = nullptr;
  STABLE_TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_current_cuda_stream(device_index, &stream));
  return static_cast<cudaStream_t>(stream);
}

inline cudaStream_t current_cuda_stream(int32_t device_index) {
  void* stream = nullptr;
  STABLE_TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_current_cuda_stream(device_index, &stream));
  return static_cast<cudaStream_t>(stream);
}

inline cublasHandle_t current_cuda_blas_handle() {
  void* handle = nullptr;
  STABLE_TORCH_ERROR_CODE_CHECK(torch_get_current_cuda_blas_handle(&handle));
  return static_cast<cublasHandle_t>(handle);
}

template <std::size_t... Indices>
auto tensor_tuple_impl(
    std::vector<torch::stable::Tensor> tensors,
    std::index_sequence<Indices...>) {
  return std::make_tuple(std::move(tensors[Indices])...);
}

template <std::size_t Size>
auto tensor_tuple(std::vector<torch::stable::Tensor> tensors) {
  return tensor_tuple_impl(std::move(tensors), std::make_index_sequence<Size>());
}

struct RecurrentDimensions {
  int64_t num_sequences;
  int64_t num_heads;
  int64_t head_size;
};

struct PreparedRecurrentMetadata {
  torch::stable::Tensor query_start_loc;
  torch::stable::Tensor state_indices;
  torch::stable::Tensor status;
  torch::stable::Tensor token_predecessor;
  torch::stable::Tensor workspace;
};

void check_cuda_contiguous(
    const torch::stable::Tensor& tensor,
    const char* name);
void check_same_device(
    const torch::stable::Tensor& reference,
    const torch::stable::Tensor& tensor,
    const char* name);
RecurrentDimensions check_recurrent_layout(
    const torch::stable::Tensor& query_start_loc,
    const torch::stable::Tensor& state_indices,
    const torch::stable::Tensor& state,
    const torch::stable::Tensor& r,
    const torch::stable::Tensor& decay_logits,
    const torch::stable::Tensor& k,
    const torch::stable::Tensor& v,
    const torch::stable::Tensor& a,
    const torch::stable::Tensor& b,
    const torch::stable::Tensor& output,
    double scale);
PreparedRecurrentMetadata prepare_recurrent_metadata_cuda(
    torch::stable::Tensor query_start_loc,
    torch::stable::Tensor state_indices,
    int64_t total_tokens,
    int64_t state_pool_size);
PreparedRecurrentMetadata prepare_recurrent_graph_metadata_cuda(
    torch::stable::Tensor query_start_loc,
    torch::stable::Tensor state_indices,
    torch::stable::Tensor num_active_tokens,
    torch::stable::Tensor num_active_sequences,
    int64_t token_capacity,
    int64_t sequence_capacity,
    int64_t state_pool_size,
    int64_t max_seqlen_capacity);

}  // namespace flashrwkv2::validation

#define FLASHRWKV_CUDA_CHECK(expression)                         \
  ::flashrwkv2::validation::check_cuda(                         \
      (expression), #expression, __FILE__, static_cast<int>(__LINE__))
