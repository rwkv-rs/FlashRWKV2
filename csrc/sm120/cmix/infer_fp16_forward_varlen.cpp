// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the FlashRWKV2 project
// Albatross source revision: ee3308f6922e59f2166c7fac3c5a192340a2b48e.

#include "../../validation.h"

#include <torch/extension.h>

#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

torch::Tensor cmix_linear_ffn_key_dispatch_f16_cuda(
    torch::Tensor x, torch::Tensor weight);
void cmix_tokenshift_forward_varlen_cuda(
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
    torch::Tensor metadata_status,
    torch::Tensor token_predecessor);
std::vector<torch::Tensor> cmix_res_ln_tokenshift_fused_forward_cuda(
    torch::Tensor x,
    torch::Tensor res,
    torch::Tensor shift_state,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor x_k,
    torch::Tensor token_predecessor,
    torch::Tensor metadata_status,
    double eps);
std::vector<torch::Tensor> post_norm_forward_varlen_cuda(
    torch::Tensor x,
    torch::Tensor res,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps,
    int64_t batch_size);
torch::Tensor cmix_relu_square_forward_varlen(torch::Tensor x);
torch::Tensor cmix_linear_ffn_down_forward_varlen(
    torch::Tensor x, torch::Tensor weight);
torch::Tensor cmix_sparse_down_relu_forward_varlen(
    torch::Tensor preact,
    torch::Tensor value_fc,
    int64_t batch_size,
    int64_t max_seqlen,
    bool deterministic);

using flashrwkv2::validation::check_cuda_contiguous;
using flashrwkv2::validation::check_same_device;
using flashrwkv2::validation::prepare_recurrent_metadata_cuda;

namespace {

void check_half(
    const torch::Tensor& tensor,
    const torch::Tensor& reference,
    const char* name) {
  check_cuda_contiguous(tensor, name);
  check_same_device(reference, tensor, name);
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat16, name, " must be float16");
}

}  // namespace

torch::Tensor cmix_tokenshift_forward_varlen(
    torch::Tensor x,
    torch::Tensor shift_state_pool,
    torch::Tensor x_k,
    torch::Tensor cu_seqlens,
    torch::Tensor state_indices,
    int64_t max_seqlen,
    py::object validated_metadata) {
  check_half(x, x, "x");
  TORCH_CHECK(x.dim() == 2 && x.size(0) > 0 && x.size(1) > 0,
              "x must have packed shape [total_tokens,C]");
  const int64_t total_tokens = x.size(0);
  const int64_t channels = x.size(1);
  check_half(shift_state_pool, x, "shift_state_pool");
  TORCH_CHECK(shift_state_pool.dim() == 2 && shift_state_pool.size(0) > 0 &&
                  shift_state_pool.size(1) == channels,
              "shift_state_pool must have shape [slots,C]");
  check_half(x_k, x, "x_k");
  TORCH_CHECK(x_k.dim() == 1 && x_k.size(0) == channels,
              "x_k must have shape [C]");
  check_cuda_contiguous(cu_seqlens, "cu_seqlens");
  check_cuda_contiguous(state_indices, "state_indices");
  check_same_device(x, cu_seqlens, "cu_seqlens");
  check_same_device(x, state_indices, "state_indices");
  TORCH_CHECK(cu_seqlens.scalar_type() == torch::kInt32 &&
                  state_indices.scalar_type() == torch::kInt32 &&
                  cu_seqlens.dim() == 1 && state_indices.dim() == 1 &&
                  state_indices.numel() > 0 &&
                  cu_seqlens.numel() == state_indices.numel() + 1,
              "invalid packed metadata");
  TORCH_CHECK(channels % 2 == 0,
              "CMix TokenShift requires an even channel count");
  const int batch_size = static_cast<int>(state_indices.numel());
  torch::Tensor launch_query_start_loc = cu_seqlens;
  torch::Tensor launch_state_indices = state_indices;
  torch::Tensor metadata_status;
  torch::Tensor token_predecessor;
  if (!validated_metadata.is_none()) {
    validated_metadata.attr("_check_compatible")(
        cu_seqlens, state_indices, total_tokens, shift_state_pool.size(0),
        max_seqlen);
    launch_query_start_loc = validated_metadata
        .attr("_query_start_loc_snapshot")()
        .cast<torch::Tensor>();
    launch_state_indices = validated_metadata
        .attr("_state_indices_snapshot")()
        .cast<torch::Tensor>();
    metadata_status = validated_metadata.attr("_status")().cast<torch::Tensor>();
    token_predecessor = validated_metadata
        .attr("_token_predecessor")()
        .cast<torch::Tensor>();
    max_seqlen = validated_metadata.attr("_max_seqlen")().cast<int64_t>();
  } else {
    if (max_seqlen <= 0) {
      max_seqlen = 1;
    }
    auto prepared = prepare_recurrent_metadata_cuda(
        cu_seqlens, state_indices, total_tokens, shift_state_pool.size(0));
    launch_query_start_loc = std::move(prepared.query_start_loc);
    launch_state_indices = std::move(prepared.state_indices);
    metadata_status = std::move(prepared.status);
    token_predecessor = std::move(prepared.token_predecessor);
  }
  auto output = torch::empty_like(x);
  cmix_tokenshift_forward_varlen_cuda(
      batch_size, static_cast<int>(total_tokens), static_cast<int>(channels),
      static_cast<int>(max_seqlen), x, shift_state_pool, x_k, output,
      launch_query_start_loc, launch_state_indices, metadata_status,
      token_predecessor);
  return output;
}

std::vector<torch::Tensor> cmix_postnorm_tokenshift_forward_varlen(
    torch::Tensor x,
    torch::Tensor res,
    torch::Tensor shift_state_pool,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor x_k,
    torch::Tensor cu_seqlens,
    torch::Tensor state_indices,
    int64_t max_seqlen,
    double eps,
    py::object validated_metadata) {
  check_half(x, x, "x");
  TORCH_CHECK(
      x.dim() == 2 && x.size(0) > 0 && x.size(1) > 0 && x.size(1) % 2 == 0,
      "CMix PostNorm TokenShift requires packed shape [total_tokens,C] with even C");
  const int64_t total_tokens = x.size(0);
  const int64_t channels = x.size(1);
  check_half(res, x, "res");
  TORCH_CHECK(res.sizes() == x.sizes(), "res shape mismatch");
  check_half(shift_state_pool, x, "shift_state_pool");
  TORCH_CHECK(
      shift_state_pool.dim() == 2 && shift_state_pool.size(0) > 0 &&
          shift_state_pool.size(1) == channels,
      "shift_state_pool must have shape [slots,C]");
  for (const auto& item : {
           std::pair<const torch::Tensor*, const char*>{&weight, "weight"},
           {&bias, "bias"},
           {&x_k, "x_k"},
       }) {
    check_half(*item.first, x, item.second);
    TORCH_CHECK(
        item.first->dim() == 1 && item.first->size(0) == channels,
        item.second, " must have shape [C]");
  }
  TORCH_CHECK(std::isfinite(eps) && eps > 0.0, "eps must be finite and positive");
  check_cuda_contiguous(cu_seqlens, "cu_seqlens");
  check_cuda_contiguous(state_indices, "state_indices");
  check_same_device(x, cu_seqlens, "cu_seqlens");
  check_same_device(x, state_indices, "state_indices");
  TORCH_CHECK(
      cu_seqlens.scalar_type() == torch::kInt32 &&
          state_indices.scalar_type() == torch::kInt32 &&
          cu_seqlens.dim() == 1 && state_indices.dim() == 1 &&
          state_indices.numel() > 0 &&
          cu_seqlens.numel() == state_indices.numel() + 1,
      "invalid packed metadata");
  const int64_t batch_size = state_indices.numel();
  torch::Tensor launch_query_start_loc = cu_seqlens;
  torch::Tensor launch_state_indices = state_indices;
  torch::Tensor metadata_status;
  torch::Tensor token_predecessor;
  if (!validated_metadata.is_none()) {
    validated_metadata.attr("_check_compatible")(
        cu_seqlens, state_indices, total_tokens, shift_state_pool.size(0),
        max_seqlen);
    launch_query_start_loc = validated_metadata
        .attr("_query_start_loc_snapshot")()
        .cast<torch::Tensor>();
    launch_state_indices = validated_metadata
        .attr("_state_indices_snapshot")()
        .cast<torch::Tensor>();
    metadata_status = validated_metadata.attr("_status")().cast<torch::Tensor>();
    token_predecessor = validated_metadata
        .attr("_token_predecessor")()
        .cast<torch::Tensor>();
    max_seqlen = validated_metadata.attr("_max_seqlen")().cast<int64_t>();
  } else {
    if (max_seqlen <= 0) {
      max_seqlen = 1;
    }
    auto prepared = prepare_recurrent_metadata_cuda(
        cu_seqlens, state_indices, total_tokens, shift_state_pool.size(0));
    launch_query_start_loc = std::move(prepared.query_start_loc);
    launch_state_indices = std::move(prepared.state_indices);
    metadata_status = std::move(prepared.status);
    token_predecessor = std::move(prepared.token_predecessor);
  }
  if (channels == 4096 && max_seqlen == 1 && total_tokens == batch_size) {
    return cmix_res_ln_tokenshift_fused_forward_cuda(
        x, res, shift_state_pool, weight, bias, x_k,
        token_predecessor, metadata_status, eps);
  }
  auto post_norm = post_norm_forward_varlen_cuda(
      x, res, weight, bias, eps, batch_size);
  auto output = torch::empty_like(x);
  cmix_tokenshift_forward_varlen_cuda(
      static_cast<int>(batch_size), static_cast<int>(total_tokens),
      static_cast<int>(channels), static_cast<int>(max_seqlen), post_norm[1],
      shift_state_pool, x_k, output, launch_query_start_loc,
      launch_state_indices, metadata_status, token_predecessor);
  return {post_norm[0], output};
}

std::vector<torch::Tensor> cmix_forward_varlen(
    torch::Tensor x,
    torch::Tensor res,
    torch::Tensor shift_state_pool,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor x_k,
    torch::Tensor key_weight,
    torch::Tensor value_weight,
    torch::Tensor cu_seqlens,
    torch::Tensor state_indices,
    int64_t max_seqlen,
    double eps,
    py::object validated_metadata,
    bool deterministic) {
  int64_t launch_max_seqlen = max_seqlen;
  if (!validated_metadata.is_none()) {
    launch_max_seqlen =
        validated_metadata.attr("_max_seqlen")().cast<int64_t>();
  }
  auto front = cmix_postnorm_tokenshift_forward_varlen(
      x, res, shift_state_pool, weight, bias, x_k, cu_seqlens,
      state_indices, max_seqlen, eps, validated_metadata);
  auto preact = cmix_linear_ffn_key_dispatch_f16_cuda(front[1], key_weight);
  torch::Tensor output;
  if (x.size(1) == 4096 && x.size(0) <= 19) {
    output = cmix_sparse_down_relu_forward_varlen(
        preact, value_weight, state_indices.numel(),
        launch_max_seqlen, deterministic);
  } else {
    output = cmix_linear_ffn_down_forward_varlen(
        cmix_relu_square_forward_varlen(preact), value_weight);
  }
  return {front[0], output};
}

void register_cmix_bindings(py::module_& module) {
  module.def(
      "cmix_forward_varlen", &cmix_forward_varlen,
      "Complete packed Albatross ChannelMix inference island",
      py::arg("x"), py::arg("res"), py::arg("shift_state_pool"),
      py::arg("weight"), py::arg("bias"), py::arg("x_k"),
      py::arg("key_weight"), py::arg("value_weight"),
      py::arg("cu_seqlens"), py::arg("state_indices"),
      py::arg("max_seqlen") = -1, py::arg("eps") = 1.0e-5,
      py::arg("validated_metadata") = py::none(),
      py::arg("deterministic") = false);
}
