// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to BlinkDL/Albatross
// Adapted from BlinkDL/Albatross commit ee3308f6922e59f2166c7fac3c5a192340a2b48e.
// The original dispatch surface is split by caller ownership here.  This
// binding keeps packed token rows and uses the CUDA implementation in the
// matching .cu file; no legacy rwkv7_v3a_ops namespace is exported.

#include "../../internal/linear/backend.cuh"

#include "validation.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

torch::stable::Tensor tmix_wkv_prepare_projection_dispatch_f16_cuda(
    torch::stable::Tensor x, torch::stable::Tensor weight);
std::vector<torch::stable::Tensor> lowrank_wag_rank_in_f16_cuda(
    torch::stable::Tensor x_w,
    torch::stable::Tensor x_a,
    torch::stable::Tensor x_g,
    std::optional<torch::stable::Tensor> w1_t,
    std::optional<torch::stable::Tensor> a1_t,
    std::optional<torch::stable::Tensor> g1_t,
    std::optional<torch::stable::Tensor> w1,
    std::optional<torch::stable::Tensor> a1,
    std::optional<torch::stable::Tensor> g1);
std::vector<torch::stable::Tensor> lowrank_wagv_rank_in_f16_cuda(
    torch::stable::Tensor x_w,
    torch::stable::Tensor x_a,
    torch::stable::Tensor x_g,
    torch::stable::Tensor x_v,
    std::optional<torch::stable::Tensor> w1_t,
    std::optional<torch::stable::Tensor> a1_t,
    std::optional<torch::stable::Tensor> g1_t,
    std::optional<torch::stable::Tensor> v1_t,
    std::optional<torch::stable::Tensor> w1,
    std::optional<torch::stable::Tensor> a1,
    std::optional<torch::stable::Tensor> g1,
    std::optional<torch::stable::Tensor> v1);
std::vector<torch::stable::Tensor> lowrank_wag_rank_out_f16_cuda(
    torch::stable::Tensor w1,
    torch::stable::Tensor a1,
    torch::stable::Tensor g1,
    std::optional<torch::stable::Tensor> w2_t,
    std::optional<torch::stable::Tensor> a2_t,
    std::optional<torch::stable::Tensor> g2_t,
    std::optional<torch::stable::Tensor> w2,
    std::optional<torch::stable::Tensor> a2,
    std::optional<torch::stable::Tensor> g2);
std::vector<torch::stable::Tensor> lowrank_wagv_vres_f16_cuda(
    torch::stable::Tensor w1,
    torch::stable::Tensor a1,
    torch::stable::Tensor g1,
    torch::stable::Tensor v1,
    std::optional<torch::stable::Tensor> w2_t,
    std::optional<torch::stable::Tensor> a2_t,
    std::optional<torch::stable::Tensor> g2_t,
    std::optional<torch::stable::Tensor> v2_t,
    std::optional<torch::stable::Tensor> w2,
    std::optional<torch::stable::Tensor> a2,
    std::optional<torch::stable::Tensor> g2,
    std::optional<torch::stable::Tensor> v2,
    torch::stable::Tensor v,
    torch::stable::Tensor v_first,
    torch::stable::Tensor v0);
void tmix_kk_a_gate_forward_varlen_cuda(
    int batch_size,
    int max_seqlen,
    int total_tokens,
    int channels,
    int heads,
    int head_size,
    torch::stable::Tensor k,
    torch::stable::Tensor k_k,
    torch::stable::Tensor a0,
    torch::stable::Tensor a12,
    torch::stable::Tensor k_a,
    torch::stable::Tensor out_k,
    torch::stable::Tensor out_a,
    torch::stable::Tensor out_b);

namespace {

void check_half(const torch::stable::Tensor& tensor, const char* name) {
  STD_TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
  STD_TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  STD_TORCH_CHECK(tensor.scalar_type() == torch::headeronly::ScalarType::Half, name, " must be float16");
}

void check_same(const torch::stable::Tensor& first, const torch::stable::Tensor& other, const char* name) {
  STD_TORCH_CHECK(other.device() == first.device(), name, " must share the input device");
}

void check_rows(const torch::stable::Tensor& tensor, const torch::stable::Tensor& reference, const char* name) {
  check_half(tensor, name);
  check_same(reference, tensor, name);
  STD_TORCH_CHECK(tensor.dim() == 2 && tensor.size(0) == reference.size(0),
              name, " must have packed shape [total_tokens,features]");
}

void check_rank_dispatch(
    const torch::stable::Tensor& x,
    const std::optional<torch::stable::Tensor>& weight,
    const std::optional<torch::stable::Tensor>& weight_t,
    bool input_projection,
    bool enforce_lowrank_limit = false) {
  check_half(x, "x");
  STD_TORCH_CHECK(x.dim() == 2 && x.size(0) > 0 && x.size(1) > 0,
              "x must have packed shape [total_tokens,features]");
  STD_TORCH_CHECK(weight.has_value() || weight_t.has_value(),
              "one of weight or weight_t must be provided");
  if (weight.has_value()) {
    check_half(*weight, "weight");
    check_same(x, *weight, "weight");
    STD_TORCH_CHECK(weight->dim() == 2 && weight->size(0) > 0 &&
                    weight->size(1) > 0,
                "weight must have rank 2");
    STD_TORCH_CHECK(
        weight->size(0) == x.size(1),
        input_projection
            ? "rank-in runtime weight must have shape [input,rank]"
            : "rank-out runtime weight must have shape [rank,output]");
  }
  if (weight_t.has_value()) {
    check_half(*weight_t, "weight_t");
    check_same(x, *weight_t, "weight_t");
    STD_TORCH_CHECK(weight_t->dim() == 2 && weight_t->size(0) > 0 &&
                    weight_t->size(1) > 0,
                "weight_t must have rank 2");
    STD_TORCH_CHECK(
        weight_t->size(1) == x.size(1),
        input_projection
            ? "rank-in original weight must have shape [rank,input]"
            : "rank-out original weight must have shape [output,rank]");
  }
  if (weight.has_value() && weight_t.has_value()) {
    const int64_t runtime_rank = input_projection ? weight->size(1) : weight->size(0);
    const int64_t original_rank = input_projection ? weight_t->size(0) : weight_t->size(1);
    STD_TORCH_CHECK(runtime_rank == original_rank,
                "weight and weight_t must describe the same rank projection");
  }
  const int64_t rank = weight.has_value()
      ? (input_projection ? weight->size(1) : weight->size(0))
      : (input_projection ? weight_t->size(0) : weight_t->size(1));
  STD_TORCH_CHECK(!enforce_lowrank_limit || rank <= 512,
              "low-rank projection requires R<=512");
}

void check_rank_out_layout(
    const torch::stable::Tensor& reference,
    int64_t rank,
    int64_t channels,
    const std::optional<torch::stable::Tensor>& weight,
    const std::optional<torch::stable::Tensor>& weight_t,
    const char* name) {
  STD_TORCH_CHECK(weight.has_value() || weight_t.has_value(),
              name, " requires an original or runtime layout");
  if (weight_t.has_value()) {
    check_half(*weight_t, name);
    check_same(reference, *weight_t, name);
    STD_TORCH_CHECK(weight_t->dim() == 2 && weight_t->size(0) == channels &&
                    weight_t->size(1) == rank,
                name, " original layout must have shape [C,R]");
  }
  if (weight.has_value()) {
    check_half(*weight, name);
    check_same(reference, *weight, name);
    STD_TORCH_CHECK(weight->dim() == 2 && weight->size(0) == rank &&
                    weight->size(1) == channels,
                name, " runtime layout must have shape [R,C]");
  }
}

}  // namespace

torch::stable::Tensor wkv_prepare_project_internal(
    torch::stable::Tensor x,
    torch::stable::Tensor weight,
    std::optional<torch::stable::Tensor> lora_a,
    std::optional<torch::stable::Tensor> lora_b,
    double lora_scale) {
  check_half(x, "x");
  check_half(weight, "weight");
  check_same(x, weight, "weight");
  STD_TORCH_CHECK(x.dim() == 2 && x.size(0) > 0 && x.size(1) > 0,
              "x must have packed shape [total_tokens,K]");
  STD_TORCH_CHECK(weight.dim() == 2 && weight.size(0) > 0 &&
                  weight.size(1) == x.size(1),
              "weight must have shape [N,K]");
  STD_TORCH_CHECK(lora_a.has_value() == lora_b.has_value(),
              "lora_a and lora_b must be provided together");
  STD_TORCH_CHECK(std::isfinite(lora_scale) &&
                  std::abs(lora_scale) <= std::numeric_limits<float>::max(),
              "lora_scale must be finite and representable as float32");
  if (!lora_a.has_value()) {
    return tmix_wkv_prepare_projection_dispatch_f16_cuda(x, weight);
  }
  check_half(*lora_a, "lora_a");
  check_half(*lora_b, "lora_b");
  check_same(x, *lora_a, "lora_a");
  check_same(x, *lora_b, "lora_b");
  STD_TORCH_CHECK(lora_a->dim() == 2 && lora_a->size(0) > 0 &&
                  lora_a->size(1) == x.size(1),
              "lora_a must have shape [R,K]");
  const int64_t rank = lora_a->size(0);
  STD_TORCH_CHECK(rank <= 512, "LoRA projection requires R<=512");
  STD_TORCH_CHECK(lora_b->dim() == 2 && lora_b->size(0) == weight.size(0) &&
                  lora_b->size(1) == rank,
              "lora_b must have shape [N,R]");
  if (static_cast<float>(lora_scale) == 0.0f) {
    return tmix_wkv_prepare_projection_dispatch_f16_cuda(x, weight);
  }
  auto output = tmix_wkv_prepare_projection_dispatch_f16_cuda(x, weight);
  return internal_linear_lora_accumulate_f16_cuda(
      x, *lora_a, *lora_b, output, lora_scale);
}

std::vector<torch::stable::Tensor> lowrank_wag_rank_in_internal(
    torch::stable::Tensor x_w,
    torch::stable::Tensor x_a,
    torch::stable::Tensor x_g,
    std::optional<torch::stable::Tensor> w1_t,
    std::optional<torch::stable::Tensor> a1_t,
    std::optional<torch::stable::Tensor> g1_t,
    std::optional<torch::stable::Tensor> w1,
    std::optional<torch::stable::Tensor> a1,
    std::optional<torch::stable::Tensor> g1) {
  check_rows(x_w, x_w, "x_w");
  check_rows(x_a, x_w, "x_a");
  check_rows(x_g, x_w, "x_g");
  check_rank_dispatch(x_w, w1, w1_t, true, true);
  check_rank_dispatch(x_a, a1, a1_t, true, true);
  check_rank_dispatch(x_g, g1, g1_t, true, true);
  return lowrank_wag_rank_in_f16_cuda(
      x_w, x_a, x_g, std::move(w1_t), std::move(a1_t),
      std::move(g1_t), std::move(w1), std::move(a1), std::move(g1));
}

std::vector<torch::stable::Tensor> lowrank_wagv_rank_in_internal(
    torch::stable::Tensor x_w,
    torch::stable::Tensor x_a,
    torch::stable::Tensor x_g,
    torch::stable::Tensor x_v,
    std::optional<torch::stable::Tensor> w1_t,
    std::optional<torch::stable::Tensor> a1_t,
    std::optional<torch::stable::Tensor> g1_t,
    std::optional<torch::stable::Tensor> v1_t,
    std::optional<torch::stable::Tensor> w1,
    std::optional<torch::stable::Tensor> a1,
    std::optional<torch::stable::Tensor> g1,
    std::optional<torch::stable::Tensor> v1) {
  check_rows(x_w, x_w, "x_w");
  check_rows(x_a, x_w, "x_a");
  check_rows(x_g, x_w, "x_g");
  check_rows(x_v, x_w, "x_v");
  check_rank_dispatch(x_w, w1, w1_t, true, true);
  check_rank_dispatch(x_a, a1, a1_t, true, true);
  check_rank_dispatch(x_g, g1, g1_t, true, true);
  check_rank_dispatch(x_v, v1, v1_t, true, true);
  return lowrank_wagv_rank_in_f16_cuda(
      x_w, x_a, x_g, x_v, std::move(w1_t), std::move(a1_t),
      std::move(g1_t), std::move(v1_t), std::move(w1), std::move(a1),
      std::move(g1), std::move(v1));
}

std::vector<torch::stable::Tensor> lowrank_wag_rank_out_internal(
    torch::stable::Tensor w1,
    torch::stable::Tensor a1,
    torch::stable::Tensor g1,
    std::optional<torch::stable::Tensor> w2_t,
    std::optional<torch::stable::Tensor> a2_t,
    std::optional<torch::stable::Tensor> g2_t,
    std::optional<torch::stable::Tensor> w2,
    std::optional<torch::stable::Tensor> a2,
    std::optional<torch::stable::Tensor> g2) {
  check_rows(w1, w1, "w1");
  check_rows(a1, w1, "a1");
  check_rows(g1, w1, "g1");
  check_rank_dispatch(w1, w2, w2_t, false, true);
  check_rank_dispatch(a1, a2, a2_t, false, true);
  check_rank_dispatch(g1, g2, g2_t, false, true);
  return lowrank_wag_rank_out_f16_cuda(
      w1, a1, g1, std::move(w2_t), std::move(a2_t), std::move(g2_t),
      std::move(w2), std::move(a2), std::move(g2));
}

std::vector<torch::stable::Tensor> lowrank_wagv_rank_out_vres_internal(
    torch::stable::Tensor w1,
    torch::stable::Tensor a1,
    torch::stable::Tensor g1,
    torch::stable::Tensor v1,
    std::optional<torch::stable::Tensor> w2_t,
    std::optional<torch::stable::Tensor> a2_t,
    std::optional<torch::stable::Tensor> g2_t,
    std::optional<torch::stable::Tensor> v2_t,
    std::optional<torch::stable::Tensor> w2,
    std::optional<torch::stable::Tensor> a2,
    std::optional<torch::stable::Tensor> g2,
    std::optional<torch::stable::Tensor> v2,
    torch::stable::Tensor v,
    torch::stable::Tensor v_first,
    torch::stable::Tensor v0) {
  for (const auto& item : {
           std::pair<const torch::stable::Tensor*, const char*>{&w1, "w1"},
           {&a1, "a1"}, {&g1, "g1"}, {&v1, "v1"},
       }) {
    check_rows(*item.first, w1, item.second);
  }
  check_rank_dispatch(w1, w2, w2_t, false, true);
  check_rank_dispatch(a1, a2, a2_t, false, true);
  check_rank_dispatch(g1, g2, g2_t, false, true);
  check_rank_dispatch(v1, v2, v2_t, false, true);
  const int64_t channels = w2_t.has_value() ? w2_t->size(0) : w2->size(1);
  check_rows(v, w1, "v");
  check_rows(v_first, v, "v_first");
  STD_TORCH_CHECK(v.size(1) == channels,
              "v and v_first must have shape [total_tokens,C]");
  check_half(v0, "v0");
  check_same(v, v0, "v0");
  STD_TORCH_CHECK(v0.dim() == 1 && v0.size(0) == v.size(1),
              "v0 must have shape [C]");
  return lowrank_wagv_vres_f16_cuda(
      w1, a1, g1, v1, std::move(w2_t), std::move(a2_t),
      std::move(g2_t), std::move(v2_t), std::move(w2), std::move(a2),
      std::move(g2), std::move(v2), v, v_first, v0);
}

std::vector<torch::stable::Tensor> wkv_prepare_wag_internal(
    torch::stable::Tensor x_w,
    torch::stable::Tensor x_a,
    torch::stable::Tensor x_g,
    std::optional<torch::stable::Tensor> w1_t,
    std::optional<torch::stable::Tensor> a1_t,
    std::optional<torch::stable::Tensor> g1_t,
    std::optional<torch::stable::Tensor> w2_t,
    std::optional<torch::stable::Tensor> a2_t,
    std::optional<torch::stable::Tensor> g2_t,
    std::optional<torch::stable::Tensor> w1,
    std::optional<torch::stable::Tensor> a1,
    std::optional<torch::stable::Tensor> g1,
    std::optional<torch::stable::Tensor> w2,
    std::optional<torch::stable::Tensor> a2,
    std::optional<torch::stable::Tensor> g2) {
  auto rank_in = lowrank_wag_rank_in_internal(
      x_w, x_a, x_g, std::move(w1_t), std::move(a1_t), std::move(g1_t),
      std::move(w1), std::move(a1), std::move(g1));
  return lowrank_wag_rank_out_internal(
      rank_in[0], rank_in[1], rank_in[2], std::move(w2_t),
      std::move(a2_t), std::move(g2_t), std::move(w2), std::move(a2),
      std::move(g2));
}

std::vector<torch::stable::Tensor> wkv_prepare_wagv_vres_internal(
    torch::stable::Tensor x_w,
    torch::stable::Tensor x_a,
    torch::stable::Tensor x_g,
    torch::stable::Tensor x_v,
    std::optional<torch::stable::Tensor> w1_t,
    std::optional<torch::stable::Tensor> a1_t,
    std::optional<torch::stable::Tensor> g1_t,
    std::optional<torch::stable::Tensor> v1_t,
    std::optional<torch::stable::Tensor> w2_t,
    std::optional<torch::stable::Tensor> a2_t,
    std::optional<torch::stable::Tensor> g2_t,
    std::optional<torch::stable::Tensor> v2_t,
    std::optional<torch::stable::Tensor> w1,
    std::optional<torch::stable::Tensor> a1,
    std::optional<torch::stable::Tensor> g1,
    std::optional<torch::stable::Tensor> v1,
    std::optional<torch::stable::Tensor> w2,
    std::optional<torch::stable::Tensor> a2,
    std::optional<torch::stable::Tensor> g2,
    std::optional<torch::stable::Tensor> v2,
    torch::stable::Tensor value,
    torch::stable::Tensor v_first,
    torch::stable::Tensor v0) {
  auto rank_in = lowrank_wagv_rank_in_internal(
      x_w, x_a, x_g, x_v, std::move(w1_t), std::move(a1_t),
      std::move(g1_t), std::move(v1_t), std::move(w1), std::move(a1),
      std::move(g1), std::move(v1));
  return lowrank_wagv_rank_out_vres_internal(
      rank_in[0], rank_in[1], rank_in[2], rank_in[3], std::move(w2_t),
      std::move(a2_t), std::move(g2_t), std::move(v2_t), std::move(w2),
      std::move(a2), std::move(g2), std::move(v2), value, v_first, v0);
}

std::vector<torch::stable::Tensor> tmix_wkv_prepare_forward_varlen(
    torch::stable::Tensor x_r,
    torch::stable::Tensor x_w,
    torch::stable::Tensor x_k,
    torch::stable::Tensor x_v,
    torch::stable::Tensor x_a,
    torch::stable::Tensor x_g,
    torch::stable::Tensor receptance_weight,
    torch::stable::Tensor key_weight,
    torch::stable::Tensor value_weight,
    std::optional<torch::stable::Tensor> w1_t,
    std::optional<torch::stable::Tensor> a1_t,
    std::optional<torch::stable::Tensor> g1_t,
    std::optional<torch::stable::Tensor> v1_t,
    std::optional<torch::stable::Tensor> w2_t,
    std::optional<torch::stable::Tensor> a2_t,
    std::optional<torch::stable::Tensor> g2_t,
    std::optional<torch::stable::Tensor> v2_t,
    torch::stable::Tensor v0,
    torch::stable::Tensor k_k,
    torch::stable::Tensor a0,
    torch::stable::Tensor k_a,
    std::optional<torch::stable::Tensor> v_first,
    std::optional<torch::stable::Tensor> w1,
    std::optional<torch::stable::Tensor> a1,
    std::optional<torch::stable::Tensor> g1,
    std::optional<torch::stable::Tensor> v1,
    std::optional<torch::stable::Tensor> w2,
    std::optional<torch::stable::Tensor> a2,
    std::optional<torch::stable::Tensor> g2,
    std::optional<torch::stable::Tensor> v2,
    std::optional<torch::stable::Tensor> receptance_lora_a,
    std::optional<torch::stable::Tensor> receptance_lora_b,
    double receptance_lora_scale,
    std::optional<torch::stable::Tensor> key_lora_a,
    std::optional<torch::stable::Tensor> key_lora_b,
    double key_lora_scale,
    std::optional<torch::stable::Tensor> value_lora_a,
    std::optional<torch::stable::Tensor> value_lora_b,
    double value_lora_scale,
    int64_t head_size,
    int64_t batch_size,
    int64_t max_seqlen) {
  for (const auto& item : {
           std::pair<const torch::stable::Tensor*, const char*>{&x_r, "x_r"},
           {&x_w, "x_w"}, {&x_k, "x_k"}, {&x_v, "x_v"},
           {&x_a, "x_a"}, {&x_g, "x_g"},
       }) {
    check_rows(*item.first, x_r, item.second);
    STD_TORCH_CHECK(item.first->sizes() == x_r.sizes(), item.second,
                " must match x_r's packed shape");
  }
  STD_TORCH_CHECK(head_size == 64 || head_size == 128 || head_size == 256,
              "head_size must be one of 64, 128, or 256");
  STD_TORCH_CHECK(x_r.size(1) % head_size == 0,
              "channels must be divisible by head_size");
  STD_TORCH_CHECK(batch_size > 0 && max_seqlen > 0,
              "batch_size and max_seqlen must be positive");
  for (const auto& item : {
           std::pair<const torch::stable::Tensor*, const char*>{
               &receptance_weight, "receptance_weight"},
           {&key_weight, "key_weight"}, {&value_weight, "value_weight"},
       }) {
    check_half(*item.first, item.second);
    check_same(x_r, *item.first, item.second);
    STD_TORCH_CHECK(item.first->dim() == 2 &&
                    item.first->size(0) == x_r.size(1) &&
                    item.first->size(1) == x_r.size(1),
                item.second, " must have shape [C,C]");
  }
  auto receptance = wkv_prepare_project_internal(
      x_r, receptance_weight, receptance_lora_a, receptance_lora_b,
      receptance_lora_scale);
  auto key = wkv_prepare_project_internal(
      x_k, key_weight, key_lora_a, key_lora_b, key_lora_scale);
  auto value = wkv_prepare_project_internal(
      x_v, value_weight, value_lora_a, value_lora_b, value_lora_scale);

  std::vector<torch::stable::Tensor> prepared;
  torch::stable::Tensor returned_v_first;
  if (v_first.has_value()) {
    prepared = wkv_prepare_wagv_vres_internal(
        x_w, x_a, x_g, x_v, w1_t, a1_t, g1_t, v1_t,
        w2_t, a2_t, g2_t, v2_t, w1, a1, g1, v1,
        w2, a2, g2, v2, value, *v_first, v0);
    value = prepared[3];
    returned_v_first = *v_first;
  } else {
    // V parameters are part of the uniform per-layer contract.  The first
    // layer validates both layouts but deliberately does not execute VRes.
    check_rank_dispatch(x_v, v1, v1_t, true, true);
    const int64_t v_rank = v1.has_value() ? v1->size(1) : v1_t->size(0);
    check_rank_out_layout(x_v, v_rank, x_v.size(1), v2, v2_t, "v2");
    prepared = wkv_prepare_wag_internal(
        x_w, x_a, x_g, w1_t, a1_t, g1_t, w2_t, a2_t, g2_t,
        w1, a1, g1, w2, a2, g2);
    returned_v_first = value;
  }

  const int64_t channels = x_r.size(1);
  for (const auto& item : {
           std::pair<const torch::stable::Tensor*, const char*>{&k_k, "k_k"},
           {&a0, "a0"}, {&k_a, "k_a"},
       }) {
    check_half(*item.first, item.second);
    check_same(x_r, *item.first, item.second);
    STD_TORCH_CHECK(item.first->dim() == 1 && item.first->size(0) == channels,
                item.second, " must have shape [C]");
  }
  auto prepared_key = torch::stable::empty_like(key);
  auto recurrent_a = torch::stable::empty_like(key);
  auto recurrent_b = torch::stable::empty_like(key);
  tmix_kk_a_gate_forward_varlen_cuda(
      static_cast<int>(batch_size), static_cast<int>(max_seqlen),
      static_cast<int>(x_r.size(0)), static_cast<int>(channels),
      static_cast<int>(channels / head_size), static_cast<int>(head_size),
      key, k_k, a0, prepared[1], k_a,
      prepared_key, recurrent_a, recurrent_b);
  return {receptance, prepared[0], prepared_key, value,
          recurrent_a, recurrent_b, prepared[2], returned_v_first};
}

auto tmix_wkv_prepare_forward_varlen_boxed(
    torch::stable::Tensor x_r, torch::stable::Tensor x_w,
    torch::stable::Tensor x_k, torch::stable::Tensor x_v,
    torch::stable::Tensor x_a, torch::stable::Tensor x_g,
    torch::stable::Tensor receptance_weight, torch::stable::Tensor key_weight,
    torch::stable::Tensor value_weight,
    std::optional<torch::stable::Tensor> w1_t,
    std::optional<torch::stable::Tensor> a1_t,
    std::optional<torch::stable::Tensor> g1_t,
    std::optional<torch::stable::Tensor> v1_t,
    std::optional<torch::stable::Tensor> w2_t,
    std::optional<torch::stable::Tensor> a2_t,
    std::optional<torch::stable::Tensor> g2_t,
    std::optional<torch::stable::Tensor> v2_t, torch::stable::Tensor v0,
    torch::stable::Tensor k_k, torch::stable::Tensor a0,
    torch::stable::Tensor k_a, std::optional<torch::stable::Tensor> v_first,
    std::optional<torch::stable::Tensor> w1,
    std::optional<torch::stable::Tensor> a1,
    std::optional<torch::stable::Tensor> g1,
    std::optional<torch::stable::Tensor> v1,
    std::optional<torch::stable::Tensor> w2,
    std::optional<torch::stable::Tensor> a2,
    std::optional<torch::stable::Tensor> g2,
    std::optional<torch::stable::Tensor> v2,
    std::optional<torch::stable::Tensor> receptance_lora_a,
    std::optional<torch::stable::Tensor> receptance_lora_b,
    double receptance_lora_scale,
    std::optional<torch::stable::Tensor> key_lora_a,
    std::optional<torch::stable::Tensor> key_lora_b, double key_lora_scale,
    std::optional<torch::stable::Tensor> value_lora_a,
    std::optional<torch::stable::Tensor> value_lora_b, double value_lora_scale,
    int64_t head_size, int64_t batch_size, int64_t max_seqlen) {
  return flashrwkv2::validation::tensor_tuple<8>(tmix_wkv_prepare_forward_varlen(
      std::move(x_r), std::move(x_w), std::move(x_k), std::move(x_v),
      std::move(x_a), std::move(x_g), std::move(receptance_weight),
      std::move(key_weight), std::move(value_weight), std::move(w1_t),
      std::move(a1_t), std::move(g1_t), std::move(v1_t), std::move(w2_t),
      std::move(a2_t), std::move(g2_t), std::move(v2_t), std::move(v0),
      std::move(k_k), std::move(a0), std::move(k_a), std::move(v_first),
      std::move(w1), std::move(a1), std::move(g1), std::move(v1),
      std::move(w2), std::move(a2), std::move(g2), std::move(v2),
      std::move(receptance_lora_a), std::move(receptance_lora_b),
      receptance_lora_scale, std::move(key_lora_a), std::move(key_lora_b),
      key_lora_scale, std::move(value_lora_a), std::move(value_lora_b),
      value_lora_scale, head_size, batch_size, max_seqlen));
}

STABLE_TORCH_LIBRARY_FRAGMENT(flashrwkv2, module) {
  module.def("tmix_wkv_prepare_forward_varlen(Tensor x_r, Tensor x_w, Tensor x_k, Tensor x_v, Tensor x_a, Tensor x_g, Tensor receptance_weight, Tensor key_weight, Tensor value_weight, Tensor? w1_t, Tensor? a1_t, Tensor? g1_t, Tensor? v1_t, Tensor? w2_t, Tensor? a2_t, Tensor? g2_t, Tensor? v2_t, Tensor v0, Tensor k_k, Tensor a0, Tensor k_a, Tensor? v_first, Tensor? w1, Tensor? a1, Tensor? g1, Tensor? v1, Tensor? w2, Tensor? a2, Tensor? g2, Tensor? v2, Tensor? receptance_lora_a, Tensor? receptance_lora_b, float receptance_lora_scale, Tensor? key_lora_a, Tensor? key_lora_b, float key_lora_scale, Tensor? value_lora_a, Tensor? value_lora_b, float value_lora_scale, int head_size, int batch_size, int max_seqlen) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
}

STABLE_TORCH_LIBRARY_IMPL(flashrwkv2, CUDA, module) {
  module.impl("tmix_wkv_prepare_forward_varlen", TORCH_BOX(&tmix_wkv_prepare_forward_varlen_boxed));
}
