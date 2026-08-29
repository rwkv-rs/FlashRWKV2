// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "validation.h"

// Native-private primitives shared by semantic operator owners. None of these
// symbols may be registered as public operators. Caller-specific dispatch is deliberately
// absent from this header and remains in TMix, CMix, or Head.
torch::stable::Tensor linear_f16_cuda(torch::stable::Tensor x, torch::stable::Tensor weight);
torch::stable::Tensor linear_f16_orig_cuda(torch::stable::Tensor x, torch::stable::Tensor weight_orig);
torch::stable::Tensor linear_orig_rows_f16_cuda(
    torch::stable::Tensor x, torch::stable::Tensor weight_orig, int64_t row_tile, int64_t out_tile);
torch::stable::Tensor linear_orig_rows_cfg_f16_cuda(
    torch::stable::Tensor x,
    torch::stable::Tensor weight_orig,
    int64_t threads,
    int64_t row_tile,
    int64_t out_tile);
torch::stable::Tensor linear_orig_rows_exact_f16_cuda(
    torch::stable::Tensor x,
    torch::stable::Tensor weight_orig,
    int64_t threads,
    int64_t out_tile,
    bool use4);
torch::stable::Tensor internal_linear_transposed_f16_cuda(
    torch::stable::Tensor x, torch::stable::Tensor weight);
torch::stable::Tensor internal_linear_lora_accumulate_f16_cuda(
    torch::stable::Tensor x,
    torch::stable::Tensor lora_a,
    torch::stable::Tensor lora_b,
    torch::stable::Tensor output,
    double lora_scale);
torch::stable::Tensor linear_f16_orig_lt_cfg_cuda(
    torch::stable::Tensor x,
    torch::stable::Tensor weight_orig,
    int64_t workspace_mb,
    int64_t algo_index);
torch::stable::Tensor linear_f16_lt_cfg_cuda(
    torch::stable::Tensor x,
    torch::stable::Tensor weight,
    int64_t workspace_mb,
    int64_t algo_index);
