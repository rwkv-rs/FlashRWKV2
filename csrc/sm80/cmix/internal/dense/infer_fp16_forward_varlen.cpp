// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the FlashRWKV2 project

#include "../../../../validation.h"

#include <torch/extension.h>

torch::Tensor cmix_linear_ffn_down_forward_varlen_cuda(
    torch::Tensor x, torch::Tensor weight);

using flashrwkv2::validation::check_cuda_contiguous;
using flashrwkv2::validation::check_same_device;

torch::Tensor cmix_linear_ffn_down_forward_varlen(
    torch::Tensor x, torch::Tensor weight) {
  check_cuda_contiguous(x, "x");
  check_cuda_contiguous(weight, "weight");
  check_same_device(x, weight, "weight");
  TORCH_CHECK(x.scalar_type() == torch::kFloat16 &&
                  weight.scalar_type() == torch::kFloat16,
              "x and weight must be float16");
  TORCH_CHECK(
      x.dim() == 2 && x.size(0) > 0 && x.size(1) > 0 &&
          weight.dim() == 2 && weight.size(0) == x.size(1) &&
          weight.size(1) > 0,
      "CMix FFN down linear expects x [rows,K] and runtime weight [K,C]");
  return cmix_linear_ffn_down_forward_varlen_cuda(x, weight);
}
