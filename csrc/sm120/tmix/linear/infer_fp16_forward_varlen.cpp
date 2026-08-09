// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to BlinkDL/Albatross
// Adapted from BlinkDL/Albatross commit ee3308f6922e59f2166c7fac3c5a192340a2b48e.
// The original dispatch surface is split by caller ownership here.  This
// binding keeps packed token rows and uses the CUDA implementation in the
// matching .cu file; no legacy rwkv7_v3a_ops namespace is exported.

#include <torch/extension.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

torch::Tensor tmix_linear_forward_varlen_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    bool weight_is_transposed,
    int64_t caller_group);
torch::Tensor tmix_linear_t_forward_varlen_cuda(
    torch::Tensor x, torch::Tensor weight_t);
torch::Tensor tmix_linear_t_tanh_forward_varlen_cuda(
    torch::Tensor x, torch::Tensor weight_t);
torch::Tensor tmix_linear_t_sigmoid_forward_varlen_cuda(
    torch::Tensor x, torch::Tensor weight_t);
torch::Tensor tmix_linear_act_tanh_forward_varlen_cuda(torch::Tensor x);
torch::Tensor tmix_linear_act_sigmoid_forward_varlen_cuda(torch::Tensor x);
torch::Tensor tmix_linear_t_vres_forward_varlen_cuda(
    torch::Tensor x,
    torch::Tensor weight_t,
    torch::Tensor v,
    torch::Tensor v_first,
    torch::Tensor v0);
torch::Tensor tmix_linear_rank_in_dispatch_forward_varlen_cuda(
    torch::Tensor x,
    std::optional<torch::Tensor> weight,
    std::optional<torch::Tensor> weight_t);
torch::Tensor tmix_linear_rank_out_dispatch_forward_varlen_cuda(
    torch::Tensor x,
    std::optional<torch::Tensor> weight,
    std::optional<torch::Tensor> weight_t);
torch::Tensor tmix_linear_attention_c2c_forward_varlen_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor lora_a,
    torch::Tensor lora_b,
    double lora_scale);
std::vector<torch::Tensor> tmix_lowrank_in_forward_varlen_cuda(
    torch::Tensor x_w,
    torch::Tensor x_a,
    torch::Tensor x_g,
    std::optional<torch::Tensor> w1_t,
    std::optional<torch::Tensor> a1_t,
    std::optional<torch::Tensor> g1_t,
    std::optional<torch::Tensor> w1,
    std::optional<torch::Tensor> a1,
    std::optional<torch::Tensor> g1);
std::vector<torch::Tensor> tmix_lowrank_wagv_in_forward_varlen_cuda(
    torch::Tensor x_w,
    torch::Tensor x_a,
    torch::Tensor x_g,
    torch::Tensor x_v,
    std::optional<torch::Tensor> w1_t,
    std::optional<torch::Tensor> a1_t,
    std::optional<torch::Tensor> g1_t,
    std::optional<torch::Tensor> v1_t,
    std::optional<torch::Tensor> w1,
    std::optional<torch::Tensor> a1,
    std::optional<torch::Tensor> g1,
    std::optional<torch::Tensor> v1);
std::vector<torch::Tensor> tmix_lowrank_out_forward_varlen_cuda(
    torch::Tensor w1,
    torch::Tensor a1,
    torch::Tensor g1,
    std::optional<torch::Tensor> w2_t,
    std::optional<torch::Tensor> a2_t,
    std::optional<torch::Tensor> g2_t,
    std::optional<torch::Tensor> w2,
    std::optional<torch::Tensor> a2,
    std::optional<torch::Tensor> g2);
std::vector<torch::Tensor> tmix_lowrank_vres_forward_varlen_cuda(
    torch::Tensor w1,
    torch::Tensor a1,
    torch::Tensor g1,
    torch::Tensor v1,
    std::optional<torch::Tensor> w2_t,
    std::optional<torch::Tensor> a2_t,
    std::optional<torch::Tensor> g2_t,
    std::optional<torch::Tensor> v2_t,
    std::optional<torch::Tensor> w2,
    std::optional<torch::Tensor> a2,
    std::optional<torch::Tensor> g2,
    std::optional<torch::Tensor> v2,
    torch::Tensor v,
    torch::Tensor v_first,
    torch::Tensor v0);

namespace {

constexpr int64_t kMaxGridDimYZ = 65535;

void check_half(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat16, name, " must be float16");
}

void check_same(const torch::Tensor& first, const torch::Tensor& other, const char* name) {
  TORCH_CHECK(other.device() == first.device(), name, " must share the input device");
}

void check_rows(const torch::Tensor& tensor, const torch::Tensor& reference, const char* name) {
  check_half(tensor, name);
  check_same(reference, tensor, name);
  TORCH_CHECK(tensor.dim() == 2 && tensor.size(0) == reference.size(0),
              name, " must have packed shape [total_tokens,features]");
}

void check_linear_t(const torch::Tensor& x, const torch::Tensor& weight_t) {
  check_half(x, "x");
  check_half(weight_t, "weight_t");
  check_same(x, weight_t, "weight_t");
  TORCH_CHECK(x.dim() == 2 && x.size(0) > 0 && x.size(1) > 0,
              "x must have packed shape [total_tokens,K]");
  TORCH_CHECK(
      x.size(0) <= kMaxGridDimYZ,
      "TMix linear_t supports at most ",
      kMaxGridDimYZ,
      " packed rows because M maps to CUDA grid.y; got ",
      x.size(0));
  TORCH_CHECK(weight_t.dim() == 2 && weight_t.size(0) > 0 &&
                  weight_t.size(1) == x.size(1),
              "weight_t must have shape [N,K]");
}

void check_rank_dispatch(
    const torch::Tensor& x,
    const std::optional<torch::Tensor>& weight,
    const std::optional<torch::Tensor>& weight_t,
    bool input_projection,
    bool enforce_lowrank_limit = false) {
  check_half(x, "x");
  TORCH_CHECK(x.dim() == 2 && x.size(0) > 0 && x.size(1) > 0,
              "x must have packed shape [total_tokens,features]");
  TORCH_CHECK(weight.has_value() || weight_t.has_value(),
              "one of weight or weight_t must be provided");
  if (weight.has_value()) {
    check_half(*weight, "weight");
    check_same(x, *weight, "weight");
    TORCH_CHECK(weight->dim() == 2 && weight->size(0) > 0 &&
                    weight->size(1) > 0,
                "weight must have rank 2");
    TORCH_CHECK(
        weight->size(0) == x.size(1),
        input_projection
            ? "rank-in runtime weight must have shape [input,rank]"
            : "rank-out runtime weight must have shape [rank,output]");
  }
  if (weight_t.has_value()) {
    check_half(*weight_t, "weight_t");
    check_same(x, *weight_t, "weight_t");
    TORCH_CHECK(weight_t->dim() == 2 && weight_t->size(0) > 0 &&
                    weight_t->size(1) > 0,
                "weight_t must have rank 2");
    TORCH_CHECK(
        weight_t->size(1) == x.size(1),
        input_projection
            ? "rank-in original weight must have shape [rank,input]"
            : "rank-out original weight must have shape [output,rank]");
  }
  if (weight.has_value() && weight_t.has_value()) {
    const int64_t runtime_rank = input_projection ? weight->size(1) : weight->size(0);
    const int64_t original_rank = input_projection ? weight_t->size(0) : weight_t->size(1);
    TORCH_CHECK(runtime_rank == original_rank,
                "weight and weight_t must describe the same rank projection");
  }
  const int64_t rank = weight.has_value()
      ? (input_projection ? weight->size(1) : weight->size(0))
      : (input_projection ? weight_t->size(0) : weight_t->size(1));
  TORCH_CHECK(!enforce_lowrank_limit || rank <= 512,
              "low-rank projection requires R<=512");
}

}  // namespace

torch::Tensor tmix_linear_caller_forward_varlen(
    torch::Tensor x,
    torch::Tensor weight,
    bool weight_is_transposed,
    int64_t caller_group) {
  check_half(x, "x");
  check_half(weight, "weight");
  check_same(x, weight, "weight");
  TORCH_CHECK(x.dim() == 2 && x.size(0) > 0 && x.size(1) > 0,
              "x must have packed shape [total_tokens,K]");
  TORCH_CHECK(weight.dim() == 2 && weight.size(0) > 0 && weight.size(1) > 0,
              "weight must have rank 2");
  if (weight_is_transposed) {
    TORCH_CHECK(weight.size(0) == x.size(1),
                "transposed weight first dimension must match x");
  } else {
    TORCH_CHECK(weight.size(1) == x.size(1),
                "weight second dimension must match x");
  }
  TORCH_CHECK(caller_group >= 0 && caller_group <= 3,
              "caller_group must be 0 (generic), 1 (att_c2c), 2 (ffn_key), or 3 (head)");
  return tmix_linear_forward_varlen_cuda(
      x, weight, weight_is_transposed, caller_group);
}

torch::Tensor tmix_linear_forward_varlen(
    torch::Tensor x, torch::Tensor weight, bool weight_is_transposed) {
  return tmix_linear_caller_forward_varlen(x, weight, weight_is_transposed, 0);
}

torch::Tensor tmix_linear_attention_c2c_forward_varlen(
    torch::Tensor x,
    torch::Tensor weight,
    std::optional<torch::Tensor> lora_a,
    std::optional<torch::Tensor> lora_b,
    double lora_scale) {
  check_half(x, "x");
  check_half(weight, "weight");
  check_same(x, weight, "weight");
  TORCH_CHECK(x.dim() == 2 && x.size(0) > 0 && x.size(1) > 0,
              "x must have packed shape [total_tokens,K]");
  TORCH_CHECK(weight.dim() == 2 && weight.size(0) > 0 &&
                  weight.size(1) == x.size(1),
              "weight must have shape [N,K]");
  TORCH_CHECK(lora_a.has_value() == lora_b.has_value(),
              "lora_a and lora_b must be provided together");
  TORCH_CHECK(std::isfinite(lora_scale) &&
                  std::abs(lora_scale) <= std::numeric_limits<float>::max(),
              "lora_scale must be finite and representable as float32");
  if (!lora_a.has_value()) {
    return tmix_linear_caller_forward_varlen(x, weight, false, 1);
  }
  check_half(*lora_a, "lora_a");
  check_half(*lora_b, "lora_b");
  check_same(x, *lora_a, "lora_a");
  check_same(x, *lora_b, "lora_b");
  TORCH_CHECK(lora_a->dim() == 2 && lora_a->size(0) > 0 &&
                  lora_a->size(1) == x.size(1),
              "lora_a must have shape [R,K]");
  const int64_t rank = lora_a->size(0);
  TORCH_CHECK(rank <= 512, "LoRA projection requires R<=512");
  TORCH_CHECK(lora_b->dim() == 2 && lora_b->size(0) == weight.size(0) &&
                  lora_b->size(1) == rank,
              "lora_b must have shape [N,R]");
  if (static_cast<float>(lora_scale) == 0.0f) {
    return tmix_linear_caller_forward_varlen(x, weight, false, 1);
  }
  return tmix_linear_attention_c2c_forward_varlen_cuda(
      x, weight, *lora_a, *lora_b, lora_scale);
}

torch::Tensor tmix_linear_ffn_key_forward_varlen(
    torch::Tensor x, torch::Tensor weight) {
  return tmix_linear_caller_forward_varlen(x, weight, false, 2);
}

torch::Tensor tmix_linear_t_forward_varlen(
    torch::Tensor x, torch::Tensor weight_t) {
  check_linear_t(x, weight_t);
  return tmix_linear_t_forward_varlen_cuda(x, weight_t);
}

torch::Tensor tmix_linear_t_tanh_forward_varlen(
    torch::Tensor x, torch::Tensor weight_t) {
  check_linear_t(x, weight_t);
  return tmix_linear_t_tanh_forward_varlen_cuda(x, weight_t);
}

torch::Tensor tmix_linear_t_sigmoid_forward_varlen(
    torch::Tensor x, torch::Tensor weight_t) {
  check_linear_t(x, weight_t);
  return tmix_linear_t_sigmoid_forward_varlen_cuda(x, weight_t);
}

torch::Tensor tmix_linear_act_tanh_forward_varlen(torch::Tensor x) {
  check_half(x, "x");
  TORCH_CHECK(x.dim() == 2 && x.size(0) > 0 && x.size(1) > 0,
              "x must have packed shape [total_tokens,features]");
  return tmix_linear_act_tanh_forward_varlen_cuda(x);
}

torch::Tensor tmix_linear_act_sigmoid_forward_varlen(torch::Tensor x) {
  check_half(x, "x");
  TORCH_CHECK(x.dim() == 2 && x.size(0) > 0 && x.size(1) > 0,
              "x must have packed shape [total_tokens,features]");
  return tmix_linear_act_sigmoid_forward_varlen_cuda(x);
}

torch::Tensor tmix_linear_t_vres_forward_varlen(
    torch::Tensor x,
    torch::Tensor weight_t,
    torch::Tensor v,
    torch::Tensor v_first,
    torch::Tensor v0) {
  check_linear_t(x, weight_t);
  check_rows(v, x, "v");
  check_rows(v_first, v, "v_first");
  check_half(v0, "v0");
  check_same(x, v0, "v0");
  TORCH_CHECK(v.size(1) == weight_t.size(0),
              "v must have shape [total_tokens,N]");
  TORCH_CHECK(v0.dim() == 1 && v0.size(0) == weight_t.size(0),
              "v0 must have shape [N]");
  return tmix_linear_t_vres_forward_varlen_cuda(
      x, weight_t, v, v_first, v0);
}

torch::Tensor tmix_linear_rank_in_dispatch_forward_varlen(
    torch::Tensor x,
    std::optional<torch::Tensor> weight,
    std::optional<torch::Tensor> weight_t) {
  check_rank_dispatch(x, weight, weight_t, true);
  return tmix_linear_rank_in_dispatch_forward_varlen_cuda(
      x, std::move(weight), std::move(weight_t));
}

torch::Tensor tmix_linear_rank_out_dispatch_forward_varlen(
    torch::Tensor x,
    std::optional<torch::Tensor> weight,
    std::optional<torch::Tensor> weight_t) {
  check_rank_dispatch(x, weight, weight_t, false);
  return tmix_linear_rank_out_dispatch_forward_varlen_cuda(
      x, std::move(weight), std::move(weight_t));
}

std::vector<torch::Tensor> tmix_lowrank_in_forward_varlen(
    torch::Tensor x_w,
    torch::Tensor x_a,
    torch::Tensor x_g,
    std::optional<torch::Tensor> w1_t,
    std::optional<torch::Tensor> a1_t,
    std::optional<torch::Tensor> g1_t,
    std::optional<torch::Tensor> w1,
    std::optional<torch::Tensor> a1,
    std::optional<torch::Tensor> g1) {
  check_rows(x_w, x_w, "x_w");
  check_rows(x_a, x_w, "x_a");
  check_rows(x_g, x_w, "x_g");
  check_rank_dispatch(x_w, w1, w1_t, true, true);
  check_rank_dispatch(x_a, a1, a1_t, true, true);
  check_rank_dispatch(x_g, g1, g1_t, true, true);
  return tmix_lowrank_in_forward_varlen_cuda(
      x_w, x_a, x_g, std::move(w1_t), std::move(a1_t),
      std::move(g1_t), std::move(w1), std::move(a1), std::move(g1));
}

std::vector<torch::Tensor> tmix_lowrank_wagv_in_forward_varlen(
    torch::Tensor x_w,
    torch::Tensor x_a,
    torch::Tensor x_g,
    torch::Tensor x_v,
    std::optional<torch::Tensor> w1_t,
    std::optional<torch::Tensor> a1_t,
    std::optional<torch::Tensor> g1_t,
    std::optional<torch::Tensor> v1_t,
    std::optional<torch::Tensor> w1,
    std::optional<torch::Tensor> a1,
    std::optional<torch::Tensor> g1,
    std::optional<torch::Tensor> v1) {
  check_rows(x_w, x_w, "x_w");
  check_rows(x_a, x_w, "x_a");
  check_rows(x_g, x_w, "x_g");
  check_rows(x_v, x_w, "x_v");
  check_rank_dispatch(x_w, w1, w1_t, true, true);
  check_rank_dispatch(x_a, a1, a1_t, true, true);
  check_rank_dispatch(x_g, g1, g1_t, true, true);
  check_rank_dispatch(x_v, v1, v1_t, true, true);
  return tmix_lowrank_wagv_in_forward_varlen_cuda(
      x_w, x_a, x_g, x_v, std::move(w1_t), std::move(a1_t),
      std::move(g1_t), std::move(v1_t), std::move(w1), std::move(a1),
      std::move(g1), std::move(v1));
}

std::vector<torch::Tensor> tmix_lowrank_out_forward_varlen(
    torch::Tensor w1,
    torch::Tensor a1,
    torch::Tensor g1,
    std::optional<torch::Tensor> w2_t,
    std::optional<torch::Tensor> a2_t,
    std::optional<torch::Tensor> g2_t,
    std::optional<torch::Tensor> w2,
    std::optional<torch::Tensor> a2,
    std::optional<torch::Tensor> g2) {
  check_rows(w1, w1, "w1");
  check_rows(a1, w1, "a1");
  check_rows(g1, w1, "g1");
  check_rank_dispatch(w1, w2, w2_t, false, true);
  check_rank_dispatch(a1, a2, a2_t, false, true);
  check_rank_dispatch(g1, g2, g2_t, false, true);
  return tmix_lowrank_out_forward_varlen_cuda(
      w1, a1, g1, std::move(w2_t), std::move(a2_t), std::move(g2_t),
      std::move(w2), std::move(a2), std::move(g2));
}

std::vector<torch::Tensor> tmix_lowrank_vres_forward_varlen(
    torch::Tensor w1,
    torch::Tensor a1,
    torch::Tensor g1,
    torch::Tensor v1,
    std::optional<torch::Tensor> w2_t,
    std::optional<torch::Tensor> a2_t,
    std::optional<torch::Tensor> g2_t,
    std::optional<torch::Tensor> v2_t,
    std::optional<torch::Tensor> w2,
    std::optional<torch::Tensor> a2,
    std::optional<torch::Tensor> g2,
    std::optional<torch::Tensor> v2,
    torch::Tensor v,
    torch::Tensor v_first,
    torch::Tensor v0) {
  for (const auto& item : {
           std::pair<const torch::Tensor*, const char*>{&w1, "w1"},
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
  TORCH_CHECK(v.size(1) == channels,
              "v and v_first must have shape [total_tokens,C]");
  check_half(v0, "v0");
  check_same(v, v0, "v0");
  TORCH_CHECK(v0.dim() == 1 && v0.size(0) == v.size(1),
              "v0 must have shape [C]");
  return tmix_lowrank_vres_forward_varlen_cuda(
      w1, a1, g1, v1, std::move(w2_t), std::move(a2_t),
      std::move(g2_t), std::move(v2_t), std::move(w2), std::move(a2),
      std::move(g2), std::move(v2), v, v_first, v0);
}

void register_tmix_linear_bindings(py::module_& module) {
  module.def(
      "tmix_linear_forward_varlen", &tmix_linear_forward_varlen,
      py::arg("x"), py::arg("weight"),
      py::arg("weight_is_transposed") = false);
  module.def(
      "tmix_linear_attention_c2c_forward_varlen",
      &tmix_linear_attention_c2c_forward_varlen,
      py::arg("x"), py::arg("weight"), py::arg("lora_a") = py::none(),
      py::arg("lora_b") = py::none(), py::arg("lora_scale") = 1.0);
  module.def(
      "tmix_linear_ffn_key_forward_varlen",
      &tmix_linear_ffn_key_forward_varlen,
      py::arg("x"), py::arg("weight"));
  module.def(
      "tmix_linear_t_forward_varlen", &tmix_linear_t_forward_varlen,
      py::arg("x"), py::arg("weight_t"));
  module.def(
      "tmix_linear_t_tanh_forward_varlen",
      &tmix_linear_t_tanh_forward_varlen,
      py::arg("x"), py::arg("weight_t"));
  module.def(
      "tmix_linear_t_sigmoid_forward_varlen",
      &tmix_linear_t_sigmoid_forward_varlen,
      py::arg("x"), py::arg("weight_t"));
  module.def(
      "tmix_linear_act_tanh_forward_varlen",
      &tmix_linear_act_tanh_forward_varlen,
      py::arg("x"));
  module.def(
      "tmix_linear_act_sigmoid_forward_varlen",
      &tmix_linear_act_sigmoid_forward_varlen,
      py::arg("x"));
  module.def(
      "tmix_linear_t_vres_forward_varlen",
      &tmix_linear_t_vres_forward_varlen,
      py::arg("x"), py::arg("weight_t"), py::arg("v"),
      py::arg("v_first"), py::arg("v0"));
  module.def(
      "tmix_linear_rank_in_dispatch_forward_varlen",
      &tmix_linear_rank_in_dispatch_forward_varlen,
      py::arg("x"), py::arg("weight") = py::none(),
      py::arg("weight_t") = py::none());
  module.def(
      "tmix_linear_rank_out_dispatch_forward_varlen",
      &tmix_linear_rank_out_dispatch_forward_varlen,
      py::arg("x"), py::arg("weight") = py::none(),
      py::arg("weight_t") = py::none());
  module.def(
      "tmix_lowrank_in_forward_varlen", &tmix_lowrank_in_forward_varlen,
      py::arg("x_w"), py::arg("x_a"), py::arg("x_g"),
      py::arg("w1_t") = py::none(), py::arg("a1_t") = py::none(),
      py::arg("g1_t") = py::none(), py::arg("w1") = py::none(),
      py::arg("a1") = py::none(), py::arg("g1") = py::none());
  module.def(
      "tmix_lowrank_wagv_in_forward_varlen",
      &tmix_lowrank_wagv_in_forward_varlen,
      py::arg("x_w"), py::arg("x_a"), py::arg("x_g"), py::arg("x_v"),
      py::arg("w1_t") = py::none(), py::arg("a1_t") = py::none(),
      py::arg("g1_t") = py::none(), py::arg("v1_t") = py::none(),
      py::arg("w1") = py::none(), py::arg("a1") = py::none(),
      py::arg("g1") = py::none(), py::arg("v1") = py::none());
  module.def(
      "tmix_lowrank_out_forward_varlen", &tmix_lowrank_out_forward_varlen,
      py::arg("w1"), py::arg("a1"), py::arg("g1"),
      py::arg("w2_t") = py::none(), py::arg("a2_t") = py::none(),
      py::arg("g2_t") = py::none(), py::arg("w2") = py::none(),
      py::arg("a2") = py::none(), py::arg("g2") = py::none());
  module.def(
      "tmix_lowrank_vres_forward_varlen", &tmix_lowrank_vres_forward_varlen,
      py::arg("w1"), py::arg("a1"), py::arg("g1"), py::arg("v1"),
      py::arg("w2_t"), py::arg("a2_t"), py::arg("g2_t"), py::arg("v2_t"),
      py::arg("w2"), py::arg("a2"), py::arg("g2"), py::arg("v2"),
      py::arg("v"), py::arg("v_first"), py::arg("v0"));
}
