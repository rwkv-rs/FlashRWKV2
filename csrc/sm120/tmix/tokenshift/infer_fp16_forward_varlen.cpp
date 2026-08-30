// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the FlashRWKV2 project
// Albatross source revision: ee3308f6922e59f2166c7fac3c5a192340a2b48e.

#include "validation.h"

#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

void tmix_tokenshift_forward_varlen(
    int batch_size,
    int total_tokens,
    int channels,
    int max_seqlen,
    torch::stable::Tensor x,
    torch::stable::Tensor shift_state,
    torch::stable::Tensor x_r,
    torch::stable::Tensor x_w,
    torch::stable::Tensor x_k,
    torch::stable::Tensor x_v,
    torch::stable::Tensor x_a,
    torch::stable::Tensor x_g,
    torch::stable::Tensor query_start_loc,
    torch::stable::Tensor state_indices,
    torch::stable::Tensor metadata_status,
    torch::stable::Tensor token_predecessor,
    std::vector<torch::stable::Tensor>& outputs);
std::vector<torch::stable::Tensor> tmix_res_ln_tokenshift_fused_forward_cuda(
    torch::stable::Tensor x,
    torch::stable::Tensor res,
    torch::stable::Tensor shift_state,
    torch::stable::Tensor weight,
    torch::stable::Tensor bias,
    torch::stable::Tensor x_r,
    torch::stable::Tensor x_w,
    torch::stable::Tensor x_k,
    torch::stable::Tensor x_v,
    torch::stable::Tensor x_a,
    torch::stable::Tensor x_g,
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

using flashrwkv2::validation::check_cuda_contiguous;

namespace {

std::vector<torch::stable::Tensor> empty_like_pack(
    const torch::stable::Tensor& reference,
    int64_t count) {
  auto storage = torch::stable::new_empty(
      reference, {count * reference.numel()});
  auto* data = storage.mutable_data_ptr<torch::headeronly::Half>();
  std::vector<torch::stable::Tensor> outputs;
  outputs.reserve(count);
  for (int64_t index = 0; index < count; ++index) {
    outputs.push_back(torch::stable::from_blob(
        data + index * reference.numel(), reference.sizes(),
        reference.strides(), reference.device(), reference.scalar_type(),
        [storage](void*) mutable {}));
  }
  return outputs;
}

void check_tensor(
    const torch::stable::Tensor& tensor,
    int32_t device_index,
    const char* name) {
  check_cuda_contiguous(tensor, name);
  STD_TORCH_CHECK(
      tensor.get_device_index() == device_index,
      name,
      " must be on the same device as x");
  STD_TORCH_CHECK(tensor.scalar_type() == torch::headeronly::ScalarType::Half, name, " must be float16");
}

}  // namespace

std::vector<torch::stable::Tensor> tmix_postnorm_tokenshift_forward_varlen(
    torch::stable::Tensor x,
    torch::stable::Tensor res,
    torch::stable::Tensor shift_state_pool,
    torch::stable::Tensor weight,
    torch::stable::Tensor bias,
    torch::stable::Tensor x_r,
    torch::stable::Tensor x_w,
    torch::stable::Tensor x_k,
    torch::stable::Tensor x_v,
    torch::stable::Tensor x_a,
    torch::stable::Tensor x_g,
    torch::stable::Tensor cu_seqlens,
    torch::stable::Tensor state_indices,
    torch::stable::Tensor metadata_status,
    int64_t max_seqlen,
    torch::stable::Tensor token_predecessor,
    double eps,
    bool) {
  check_cuda_contiguous(x, "x");
  const auto device_index = x.get_device_index();
  STD_TORCH_CHECK(
      x.scalar_type() == torch::headeronly::ScalarType::Half,
      "x must be float16");
  const auto x_sizes = x.sizes();
  STD_TORCH_CHECK(
      x_sizes.size() == 2 && x_sizes[0] > 0 && x_sizes[1] > 0 &&
          x_sizes[1] % 2 == 0,
      "TMix PostNorm TokenShift requires packed shape [total_tokens,C] with even C");
  const int64_t total_tokens = x_sizes[0];
  const int64_t channels = x_sizes[1];
  check_tensor(res, device_index, "res");
  STD_TORCH_CHECK(res.sizes() == x_sizes, "res shape mismatch");
  check_tensor(shift_state_pool, device_index, "shift_state_pool");
  const auto shift_state_sizes = shift_state_pool.sizes();
  STD_TORCH_CHECK(
      shift_state_sizes.size() == 2 && shift_state_sizes[0] > 0 &&
          shift_state_sizes[1] == channels,
      "shift_state_pool must have shape [slots,C]");
  for (const auto& item : {
           std::pair<const torch::stable::Tensor*, const char*>{&weight, "weight"},
           {&bias, "bias"}, {&x_r, "x_r"}, {&x_w, "x_w"},
           {&x_k, "x_k"}, {&x_v, "x_v"}, {&x_a, "x_a"}, {&x_g, "x_g"},
       }) {
    check_tensor(*item.first, device_index, item.second);
    const auto sizes = item.first->sizes();
    STD_TORCH_CHECK(
        sizes.size() == 1 && sizes[0] == channels,
        item.second, " must have shape [C]");
  }
  STD_TORCH_CHECK(std::isfinite(eps) && eps > 0.0, "eps must be finite and positive");
  check_cuda_contiguous(cu_seqlens, "cu_seqlens");
  check_cuda_contiguous(state_indices, "state_indices");
  STD_TORCH_CHECK(
      cu_seqlens.get_device_index() == device_index,
      "cu_seqlens must be on the same device as x");
  STD_TORCH_CHECK(
      state_indices.get_device_index() == device_index,
      "state_indices must be on the same device as x");
  const auto cu_seqlens_sizes = cu_seqlens.sizes();
  const auto state_indices_sizes = state_indices.sizes();
  STD_TORCH_CHECK(
      cu_seqlens.scalar_type() == torch::headeronly::ScalarType::Int &&
          state_indices.scalar_type() == torch::headeronly::ScalarType::Int &&
          cu_seqlens_sizes.size() == 1 && state_indices_sizes.size() == 1 &&
          state_indices.numel() > 0 &&
          cu_seqlens.numel() == state_indices.numel() + 1,
      "invalid packed metadata");
  const int64_t batch_size = state_indices.numel();
  if (batch_size == 1 && channels == 4096 && total_tokens == 1 &&
      max_seqlen == 1) {
    return tmix_res_ln_tokenshift_fused_forward_cuda(
        x, res, shift_state_pool, weight, bias, x_r, x_w, x_k, x_v,
        x_a, x_g, token_predecessor, metadata_status, eps);
  }
  auto post_norm = post_norm_forward_varlen_cuda(
      x, res, weight, bias, eps, batch_size);
  auto shifted = empty_like_pack(x, 6);
  tmix_tokenshift_forward_varlen(
      static_cast<int>(batch_size), static_cast<int>(total_tokens),
      static_cast<int>(channels), static_cast<int>(max_seqlen), post_norm[1],
      shift_state_pool, x_r, x_w, x_k, x_v, x_a, x_g,
      cu_seqlens, state_indices, metadata_status,
      token_predecessor, shifted);
  std::vector<torch::stable::Tensor> outputs;
  outputs.reserve(7);
  outputs.push_back(post_norm[0]);
  outputs.insert(outputs.end(), shifted.begin(), shifted.end());
  return outputs;
}

std::tuple<torch::stable::Tensor, torch::stable::Tensor,
           torch::stable::Tensor, torch::stable::Tensor,
           torch::stable::Tensor, torch::stable::Tensor,
           torch::stable::Tensor>
tmix_postnorm_tokenshift_forward_varlen_boxed(
    torch::stable::Tensor x, torch::stable::Tensor res,
    torch::stable::Tensor shift_state_pool, torch::stable::Tensor weight,
    torch::stable::Tensor bias, torch::stable::Tensor x_r,
    torch::stable::Tensor x_w, torch::stable::Tensor x_k,
    torch::stable::Tensor x_v, torch::stable::Tensor x_a,
    torch::stable::Tensor x_g, torch::stable::Tensor cu_seqlens,
    torch::stable::Tensor state_indices, torch::stable::Tensor metadata_status,
    int64_t max_seqlen, torch::stable::Tensor token_predecessor, double eps) {
  return flashrwkv2::validation::tensor_tuple<7>(
      tmix_postnorm_tokenshift_forward_varlen(
          std::move(x), std::move(res), std::move(shift_state_pool),
          std::move(weight), std::move(bias), std::move(x_r), std::move(x_w),
          std::move(x_k), std::move(x_v), std::move(x_a), std::move(x_g),
          std::move(cu_seqlens), std::move(state_indices),
          std::move(metadata_status), max_seqlen, std::move(token_predecessor),
          eps, false));
}

STABLE_TORCH_LIBRARY_FRAGMENT(flashrwkv2, module) {
  module.def("tmix_postnorm_tokenshift_forward_varlen(Tensor x, Tensor res, Tensor(a!) shift_state_pool, Tensor weight, Tensor bias, Tensor x_r, Tensor x_w, Tensor x_k, Tensor x_v, Tensor x_a, Tensor x_g, Tensor cu_seqlens, Tensor state_indices, Tensor metadata_status, int max_seqlen, Tensor token_predecessor, float eps) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
}

STABLE_TORCH_LIBRARY_IMPL(flashrwkv2, CUDA, module) {
  module.impl("tmix_postnorm_tokenshift_forward_varlen", TORCH_BOX(&tmix_postnorm_tokenshift_forward_varlen_boxed));
}
