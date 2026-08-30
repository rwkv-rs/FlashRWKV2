// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <ATen/ATen.h>

// Native-private primitives shared by semantic operator owners. None of these
// symbols may be registered with pybind. Caller-specific dispatch is deliberately
// absent from this header and remains in TMix, CMix, or Head.
at::Tensor linear_f16_cuda(at::Tensor x, at::Tensor weight);
at::Tensor linear_f16_orig_cuda(at::Tensor x, at::Tensor weight_orig);
at::Tensor linear_orig_rows_f16_cuda(
    at::Tensor x, at::Tensor weight_orig, int64_t row_tile, int64_t out_tile);
at::Tensor linear_orig_rows_cfg_f16_cuda(
    at::Tensor x,
    at::Tensor weight_orig,
    int64_t threads,
    int64_t row_tile,
    int64_t out_tile);
at::Tensor linear_orig_rows_exact_f16_cuda(
    at::Tensor x,
    at::Tensor weight_orig,
    int64_t threads,
    int64_t out_tile,
    bool use4);
at::Tensor internal_linear_transposed_f16_cuda(
    at::Tensor x, at::Tensor weight);
at::Tensor internal_linear_lora_accumulate_f16_cuda(
    at::Tensor x,
    at::Tensor lora_a,
    at::Tensor lora_b,
    at::Tensor output,
    double lora_scale);
at::Tensor linear_f16_orig_lt_cfg_cuda(
    at::Tensor x,
    at::Tensor weight_orig,
    int64_t workspace_mb,
    int64_t algo_index);
at::Tensor linear_f16_lt_cfg_cuda(
    at::Tensor x,
    at::Tensor weight,
    int64_t workspace_mb,
    int64_t algo_index);
