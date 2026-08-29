// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the FlashRWKV2 project

#include "../../../../validation.h"

#include "validation.h"

torch::stable::Tensor cmix_relu_square_forward_varlen_cuda(torch::stable::Tensor x);

using flashrwkv2::validation::check_cuda_contiguous;

torch::stable::Tensor cmix_relu_square_forward_varlen(torch::stable::Tensor x) {
  check_cuda_contiguous(x, "x");
  STD_TORCH_CHECK(x.scalar_type() == torch::headeronly::ScalarType::Half, "x must be float16");
  STD_TORCH_CHECK(x.dim() == 2 && x.size(0) > 0 && x.size(1) > 0,
              "x must have packed shape [total_tokens,features]");
  STD_TORCH_CHECK((x.numel() % 2) == 0,
              "cmix relu-square requires an even number of elements");
  return cmix_relu_square_forward_varlen_cuda(x);
}
