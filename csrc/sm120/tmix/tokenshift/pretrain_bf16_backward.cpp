// SPDX-License-Identifier: Apache-2.0
// RWKV-LM train_temp revision: 952102498e9ed367ea0a59ee64106916d474d30f.

#include <torch/extension.h>

#include <utility>
#include <vector>

std::vector<torch::Tensor> pretrain_tmix_tokenshift_backward_cuda(
    torch::Tensor grad_r, torch::Tensor grad_w, torch::Tensor grad_k,
    torch::Tensor grad_v, torch::Tensor grad_a, torch::Tensor grad_g,
    torch::Tensor x, torch::Tensor x_r, torch::Tensor x_w, torch::Tensor x_k,
    torch::Tensor x_v, torch::Tensor x_a, torch::Tensor x_g);

namespace {
void check_bf16_cuda(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous() &&
                  tensor.scalar_type() == torch::kBFloat16,
              name, " must be contiguous CUDA bfloat16");
}
}  // namespace

std::vector<torch::Tensor> pretrain_tmix_tokenshift_backward(
    torch::Tensor grad_r, torch::Tensor grad_w, torch::Tensor grad_k,
    torch::Tensor grad_v, torch::Tensor grad_a, torch::Tensor grad_g,
    torch::Tensor x, torch::Tensor x_r, torch::Tensor x_w, torch::Tensor x_k,
    torch::Tensor x_v, torch::Tensor x_a, torch::Tensor x_g) {
  for (const auto& item : {
           std::pair<const torch::Tensor*, const char*>{&grad_r, "grad_r"},
           {&grad_w, "grad_w"}, {&grad_k, "grad_k"}, {&grad_v, "grad_v"},
           {&grad_a, "grad_a"}, {&grad_g, "grad_g"}, {&x, "x"},
           {&x_r, "x_r"}, {&x_w, "x_w"}, {&x_k, "x_k"}, {&x_v, "x_v"},
           {&x_a, "x_a"}, {&x_g, "x_g"}}) {
    check_bf16_cuda(*item.first, item.second);
  }
  TORCH_CHECK(x.dim() == 3 && grad_r.sizes() == x.sizes() &&
                  grad_w.sizes() == x.sizes() && grad_k.sizes() == x.sizes() &&
                  grad_v.sizes() == x.sizes() && grad_a.sizes() == x.sizes() &&
                  grad_g.sizes() == x.sizes(),
              "tokenshift gradient shapes must match x");
  TORCH_CHECK(x.device() == grad_r.device() && x_r.device() == x.device() &&
                  x_w.device() == x.device() && x_k.device() == x.device() &&
                  x_v.device() == x.device() && x_a.device() == x.device() &&
                  x_g.device() == x.device(),
              "tokenshift tensors must share a device");
  return pretrain_tmix_tokenshift_backward_cuda(
      grad_r, grad_w, grad_k, grad_v, grad_a, grad_g, x, x_r, x_w, x_k, x_v,
      x_a, x_g);
}

void register_pretrain_tmix_tokenshift_backward_bindings(py::module_& module) {
  module.def("pretrain_tmix_tokenshift_backward", &pretrain_tmix_tokenshift_backward,
             py::arg("grad_r"), py::arg("grad_w"), py::arg("grad_k"),
             py::arg("grad_v"), py::arg("grad_a"), py::arg("grad_g"),
             py::arg("x"), py::arg("x_r"), py::arg("x_w"), py::arg("x_k"),
             py::arg("x_v"), py::arg("x_a"), py::arg("x_g"));
}
