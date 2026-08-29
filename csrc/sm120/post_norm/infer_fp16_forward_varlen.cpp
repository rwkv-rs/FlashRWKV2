// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to BlinkDL/Albatross
// Adapted from BlinkDL/Albatross commit ee3308f6922e59f2166c7fac3c5a192340a2b48e.
// This binding file exposes only the final-output PostNorm island.  Internal
// PostNorm launchers are linked directly by their TMix and CMix owners.

#include "validation.h"

#include <cmath>
torch::stable::Tensor post_norm_output_forward_varlen_cuda(
    torch::stable::Tensor x,
    torch::stable::Tensor res,
    torch::stable::Tensor weight,
    torch::stable::Tensor bias,
    double eps);
namespace {

void check_half(const torch::stable::Tensor& tensor, const char* name) {
  STD_TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
  STD_TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  STD_TORCH_CHECK(tensor.scalar_type() == torch::headeronly::ScalarType::Half, name, " must be float16");
}

void check_rows(const torch::stable::Tensor& tensor, const torch::stable::Tensor& reference, const char* name) {
  check_half(tensor, name);
  STD_TORCH_CHECK(tensor.device() == reference.device(), name, " must share the input device");
  STD_TORCH_CHECK(tensor.sizes() == reference.sizes(), name, " shape mismatch");
}

void check_affine(const torch::stable::Tensor& x, const torch::stable::Tensor& weight, const torch::stable::Tensor& bias) {
  check_half(weight, "weight");
  check_half(bias, "bias");
  STD_TORCH_CHECK(weight.device() == x.device() && bias.device() == x.device(),
              "LN parameters must share x's device");
  STD_TORCH_CHECK(weight.dim() == 1 && bias.dim() == 1 &&
                  weight.size(0) == x.size(-1) && bias.size(0) == x.size(-1),
              "weight and bias must have shape [C]");
}

void check_eps(double eps) {
  STD_TORCH_CHECK(std::isfinite(eps) && eps > 0.0, "eps must be finite and positive");
}

}  // namespace

torch::stable::Tensor post_norm_output_forward_varlen(
    torch::stable::Tensor x,
    torch::stable::Tensor res,
    torch::stable::Tensor weight,
    torch::stable::Tensor bias,
    double eps) {
  check_half(x, "x");
  STD_TORCH_CHECK(x.dim() == 2 && x.size(0) > 0 && x.size(1) > 0,
              "x must have packed shape [total_tokens,C]");
  check_rows(res, x, "res");
  check_affine(x, weight, bias);
  check_eps(eps);
  return post_norm_output_forward_varlen_cuda(x, res, weight, bias, eps);
}

STABLE_TORCH_LIBRARY_FRAGMENT(flashrwkv2, module) {
  module.def("post_norm_output_forward_varlen(Tensor x, Tensor res, Tensor weight, Tensor bias, float eps) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(flashrwkv2, CUDA, module) {
  module.impl("post_norm_output_forward_varlen", TORCH_BOX(&post_norm_output_forward_varlen));
}
