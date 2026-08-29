// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the FlashRWKV2 project
// Albatross source revision: ee3308f6922e59f2166c7fac3c5a192340a2b48e.

#include "validation.h"

#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

torch::stable::Tensor cmix_linear_ffn_key_dispatch_f16_cuda(
    torch::stable::Tensor x, torch::stable::Tensor weight);
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
    torch::stable::Tensor token_predecessor);
std::vector<torch::stable::Tensor> cmix_res_ln_tokenshift_fused_forward_cuda(
    torch::stable::Tensor x,
    torch::stable::Tensor res,
    torch::stable::Tensor shift_state,
    torch::stable::Tensor weight,
    torch::stable::Tensor bias,
    torch::stable::Tensor x_k,
    torch::stable::Tensor token_predecessor,
    torch::stable::Tensor metadata_status,
    double eps);
std::vector<torch::stable::Tensor> post_norm_forward_varlen_cuda(
    torch::stable::Tensor x,
    torch::stable::Tensor res,
    torch::stable::Tensor weight,
    torch::stable::Tensor bias,
    double eps,
    int64_t batch_size);
torch::stable::Tensor cmix_relu_square_forward_varlen(torch::stable::Tensor x);
torch::stable::Tensor cmix_linear_ffn_down_forward_varlen(
    torch::stable::Tensor x, torch::stable::Tensor weight);
torch::stable::Tensor cmix_sparse_down_relu_forward_varlen(
    torch::stable::Tensor preact,
    torch::stable::Tensor value_fc,
    int64_t batch_size,
    int64_t max_seqlen,
    bool deterministic);

using flashrwkv2::validation::check_cuda_contiguous;
using flashrwkv2::validation::check_same_device;

namespace {

void check_half(
    const torch::stable::Tensor& tensor,
    const torch::stable::Tensor& reference,
    const char* name) {
  check_cuda_contiguous(tensor, name);
  check_same_device(reference, tensor, name);
  STD_TORCH_CHECK(tensor.scalar_type() == torch::headeronly::ScalarType::Half, name, " must be float16");
}

}  // namespace

torch::stable::Tensor cmix_tokenshift_forward_varlen(
    torch::stable::Tensor x,
    torch::stable::Tensor shift_state_pool,
    torch::stable::Tensor x_k,
    torch::stable::Tensor cu_seqlens,
    torch::stable::Tensor state_indices,
    torch::stable::Tensor metadata_status,
    int64_t max_seqlen,
    torch::stable::Tensor token_predecessor) {
  check_half(x, x, "x");
  STD_TORCH_CHECK(x.dim() == 2 && x.size(0) > 0 && x.size(1) > 0,
              "x must have packed shape [total_tokens,C]");
  const int64_t total_tokens = x.size(0);
  const int64_t channels = x.size(1);
  check_half(shift_state_pool, x, "shift_state_pool");
  STD_TORCH_CHECK(shift_state_pool.dim() == 2 && shift_state_pool.size(0) > 0 &&
                  shift_state_pool.size(1) == channels,
              "shift_state_pool must have shape [slots,C]");
  check_half(x_k, x, "x_k");
  STD_TORCH_CHECK(x_k.dim() == 1 && x_k.size(0) == channels,
              "x_k must have shape [C]");
  check_cuda_contiguous(cu_seqlens, "cu_seqlens");
  check_cuda_contiguous(state_indices, "state_indices");
  check_same_device(x, cu_seqlens, "cu_seqlens");
  check_same_device(x, state_indices, "state_indices");
  STD_TORCH_CHECK(cu_seqlens.scalar_type() == torch::headeronly::ScalarType::Int &&
                  state_indices.scalar_type() == torch::headeronly::ScalarType::Int &&
                  cu_seqlens.dim() == 1 && state_indices.dim() == 1 &&
                  state_indices.numel() > 0 &&
                  cu_seqlens.numel() == state_indices.numel() + 1,
              "invalid packed metadata");
  STD_TORCH_CHECK(channels % 2 == 0,
              "CMix TokenShift requires an even channel count");
  const int batch_size = static_cast<int>(state_indices.numel());
  auto output = torch::stable::empty_like(x);
  cmix_tokenshift_forward_varlen_cuda(
      batch_size, static_cast<int>(total_tokens), static_cast<int>(channels),
      static_cast<int>(max_seqlen), x, shift_state_pool, x_k, output,
      cu_seqlens, state_indices, metadata_status,
      token_predecessor);
  return output;
}

std::vector<torch::stable::Tensor> cmix_postnorm_tokenshift_forward_varlen(
    torch::stable::Tensor x,
    torch::stable::Tensor res,
    torch::stable::Tensor shift_state_pool,
    torch::stable::Tensor weight,
    torch::stable::Tensor bias,
    torch::stable::Tensor x_k,
    torch::stable::Tensor cu_seqlens,
    torch::stable::Tensor state_indices,
    torch::stable::Tensor metadata_status,
    int64_t max_seqlen,
    torch::stable::Tensor token_predecessor,
    double eps) {
  check_half(x, x, "x");
  STD_TORCH_CHECK(
      x.dim() == 2 && x.size(0) > 0 && x.size(1) > 0 && x.size(1) % 2 == 0,
      "CMix PostNorm TokenShift requires packed shape [total_tokens,C] with even C");
  const int64_t total_tokens = x.size(0);
  const int64_t channels = x.size(1);
  check_half(res, x, "res");
  STD_TORCH_CHECK(res.sizes() == x.sizes(), "res shape mismatch");
  check_half(shift_state_pool, x, "shift_state_pool");
  STD_TORCH_CHECK(
      shift_state_pool.dim() == 2 && shift_state_pool.size(0) > 0 &&
          shift_state_pool.size(1) == channels,
      "shift_state_pool must have shape [slots,C]");
  for (const auto& item : {
           std::pair<const torch::stable::Tensor*, const char*>{&weight, "weight"},
           {&bias, "bias"},
           {&x_k, "x_k"},
       }) {
    check_half(*item.first, x, item.second);
    STD_TORCH_CHECK(
        item.first->dim() == 1 && item.first->size(0) == channels,
        item.second, " must have shape [C]");
  }
  STD_TORCH_CHECK(std::isfinite(eps) && eps > 0.0, "eps must be finite and positive");
  check_cuda_contiguous(cu_seqlens, "cu_seqlens");
  check_cuda_contiguous(state_indices, "state_indices");
  check_same_device(x, cu_seqlens, "cu_seqlens");
  check_same_device(x, state_indices, "state_indices");
  STD_TORCH_CHECK(
      cu_seqlens.scalar_type() == torch::headeronly::ScalarType::Int &&
          state_indices.scalar_type() == torch::headeronly::ScalarType::Int &&
          cu_seqlens.dim() == 1 && state_indices.dim() == 1 &&
          state_indices.numel() > 0 &&
          cu_seqlens.numel() == state_indices.numel() + 1,
      "invalid packed metadata");
  const int64_t batch_size = state_indices.numel();
  if (channels == 4096 && max_seqlen == 1 && total_tokens == batch_size) {
    return cmix_res_ln_tokenshift_fused_forward_cuda(
        x, res, shift_state_pool, weight, bias, x_k,
        token_predecessor, metadata_status, eps);
  }
  auto post_norm = post_norm_forward_varlen_cuda(
      x, res, weight, bias, eps, batch_size);
  auto output = torch::stable::empty_like(x);
  cmix_tokenshift_forward_varlen_cuda(
      static_cast<int>(batch_size), static_cast<int>(total_tokens),
      static_cast<int>(channels), static_cast<int>(max_seqlen), post_norm[1],
      shift_state_pool, x_k, output, cu_seqlens,
      state_indices, metadata_status, token_predecessor);
  return {post_norm[0], output};
}

std::tuple<torch::stable::Tensor, torch::stable::Tensor> cmix_forward_varlen(
    torch::stable::Tensor x,
    torch::stable::Tensor res,
    torch::stable::Tensor shift_state_pool,
    torch::stable::Tensor weight,
    torch::stable::Tensor bias,
    torch::stable::Tensor x_k,
    torch::stable::Tensor key_weight,
    torch::stable::Tensor value_weight,
    torch::stable::Tensor cu_seqlens,
    torch::stable::Tensor state_indices,
    torch::stable::Tensor metadata_status,
    int64_t max_seqlen,
    torch::stable::Tensor token_predecessor,
    double eps,
    bool deterministic) {
  auto front = cmix_postnorm_tokenshift_forward_varlen(
      x, res, shift_state_pool, weight, bias, x_k, cu_seqlens,
      state_indices, metadata_status, max_seqlen, token_predecessor, eps);
  auto preact = cmix_linear_ffn_key_dispatch_f16_cuda(front[1], key_weight);
  torch::stable::Tensor output;
  if (x.size(1) == 4096 && x.size(0) <= 19) {
    output = cmix_sparse_down_relu_forward_varlen(
        preact, value_weight, state_indices.numel(),
        max_seqlen, deterministic);
  } else {
    output = cmix_linear_ffn_down_forward_varlen(
        cmix_relu_square_forward_varlen(preact), value_weight);
  }
  return std::make_tuple(std::move(front[0]), std::move(output));
}

STABLE_TORCH_LIBRARY_FRAGMENT(flashrwkv2, module) {
  module.def("cmix_forward_varlen(Tensor x, Tensor res, Tensor(a!) shift_state_pool, Tensor weight, Tensor bias, Tensor x_k, Tensor key_weight, Tensor value_weight, Tensor cu_seqlens, Tensor state_indices, Tensor metadata_status, int max_seqlen, Tensor token_predecessor, float eps, bool deterministic) -> (Tensor, Tensor)");
}

STABLE_TORCH_LIBRARY_IMPL(flashrwkv2, CUDA, module) {
  module.impl("cmix_forward_varlen", TORCH_BOX(&cmix_forward_varlen));
}
