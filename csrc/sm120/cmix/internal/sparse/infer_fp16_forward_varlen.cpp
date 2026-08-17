// SPDX-License-Identifier: Apache-2.0
// Native-private sparse CMix down-projection owner.

#include "../../../../validation.h"

#include <torch/extension.h>

torch::Tensor cmix_sparse_down_relu_forward_varlen_cuda(
    torch::Tensor preact,
    torch::Tensor value_fc,
    int64_t batch_size,
    int64_t max_seqlen,
    bool deterministic);

using flashrwkv2::validation::check_cuda_contiguous;
using flashrwkv2::validation::check_same_device;

torch::Tensor cmix_sparse_down_relu_forward_varlen(
    torch::Tensor preact,
    torch::Tensor value_fc,
    int64_t batch_size,
    int64_t max_seqlen,
    bool deterministic) {
  check_cuda_contiguous(preact, "preact");
  check_cuda_contiguous(value_fc, "value_fc");
  check_same_device(preact, value_fc, "value_fc");
  TORCH_CHECK(preact.scalar_type() == torch::kFloat16 &&
                  value_fc.scalar_type() == torch::kFloat16,
              "preact and value_fc must be float16");
  TORCH_CHECK(preact.dim() == 2 && value_fc.dim() == 2 &&
                  preact.size(1) == value_fc.size(0),
              "sparse down expects preact [rows,F] and value_fc [F,C]");
  TORCH_CHECK(batch_size > 0 && max_seqlen > 0,
              "batch_size and max_seqlen must be positive");
  return cmix_sparse_down_relu_forward_varlen_cuda(
      preact, value_fc, batch_size, max_seqlen, deterministic);
}
