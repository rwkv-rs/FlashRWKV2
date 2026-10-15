// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <ATen/ATen.h>
#include <vector>

void internal_linear_train_check_adapter(
    const at::Tensor& x, const at::Tensor& weight,
    const at::Tensor& a, const at::Tensor& b, double scale);
void internal_linear_train_check(
    const at::Tensor& x, const at::Tensor& weight,
    const at::Tensor& a, const at::Tensor& b, double scale);
void internal_linear_train_check_backward(
    const at::Tensor& grad, const at::Tensor& x, const at::Tensor& weight);
at::Tensor internal_linear_train_normalize_backward(at::Tensor a, at::Tensor grad);
at::Tensor internal_linear_train_gemm(
    const at::Tensor& left, const at::Tensor& right,
    bool transpose_left = false, bool transpose_right = false,
    double scale = 1.0, bool fp32_output = false,
    at::Tensor output = {}, double beta = 0.0);
std::vector<at::Tensor> internal_linear_train_forward(
    at::Tensor x, at::Tensor weight, at::Tensor a, at::Tensor b,
    double scale, bool use_nora);
std::vector<at::Tensor> internal_linear_train_backward(
    at::Tensor grad, at::Tensor x, at::Tensor weight,
    at::Tensor a, at::Tensor b, at::Tensor features, at::Tensor normalized_a,
    double scale, bool use_nora, bool need_x, bool need_weight,
    bool need_a, bool need_b);

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
