// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright contributors to Rapid-Sampling
// Adapted from Rapid-Sampling revision e0297f7830c3fa581d49ddddddba32f35ea7f733
// and rwkv-rs/vllm-rwkv revision fd440426689f10e240b5761e1a7c82e4c37deb8d.

#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>

#include <cmath>
#include <limits>
#include <optional>

int64_t sampling_state_size_cuda();
torch::Tensor setup_sampling_states_cuda(int64_t seed, int64_t num_slots);
torch::Tensor sampling_temperature_topk_topp_scalar_cuda(
    torch::Tensor logits, torch::Tensor states, torch::Tensor slot_indices,
    double temperature, int64_t top_k, double top_p,
    torch::Tensor num_active_samples);
torch::Tensor sampling_temperature_topk_topp_per_request_cuda(
    torch::Tensor logits, torch::Tensor states, torch::Tensor slot_indices,
    torch::Tensor temperatures, torch::Tensor top_ks, torch::Tensor top_ps,
    torch::Tensor num_active_samples);
torch::Tensor sampling_six_parameter_scalar_cuda(
    torch::Tensor logits, torch::Tensor penalties, torch::Tensor states,
    torch::Tensor slot_indices, double presence_penalty,
    double frequency_penalty, double penalty_decay, double temperature,
    int64_t top_k, double top_p, torch::Tensor num_active_samples);
torch::Tensor sampling_six_parameter_per_request_cuda(
    torch::Tensor logits, torch::Tensor penalties, torch::Tensor states,
    torch::Tensor slot_indices, torch::Tensor presence_penalties,
    torch::Tensor frequency_penalties, torch::Tensor penalty_decays,
    torch::Tensor temperatures, torch::Tensor top_ks, torch::Tensor top_ps,
    torch::Tensor num_active_samples);

namespace {

void check_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_logits(const torch::Tensor& logits) {
  check_cuda_contiguous(logits, "logits");
  TORCH_CHECK(logits.scalar_type() == torch::kFloat32,
              "logits must be float32");
  TORCH_CHECK(logits.dim() == 2 && logits.size(0) > 0,
              "logits must have compact request shape [B,V]");
  const int64_t vocab_size = logits.size(1);
  TORCH_CHECK(vocab_size > 0 && vocab_size <= 1048576 && vocab_size % 4 == 0,
              "vocabulary size must be positive, divisible by 4, and no larger "
              "than 1048576");
  TORCH_CHECK(logits.size(0) <= std::numeric_limits<int>::max() &&
                  vocab_size <= std::numeric_limits<int>::max(),
              "sampling dimensions must fit int32");
}

void check_state_and_slots(const torch::Tensor& logits,
                           const torch::Tensor& states,
                           const torch::Tensor& slot_indices) {
  check_cuda_contiguous(states, "states");
  TORCH_CHECK(states.scalar_type() == torch::kInt8,
              "states must be int8 sampling state storage");
  TORCH_CHECK(states.dim() == 2 && states.size(0) > 0 &&
                  states.size(1) == sampling_state_size_cuda(),
              "states must come from setup_sampling_states");
  check_cuda_contiguous(slot_indices, "slot_indices");
  TORCH_CHECK(slot_indices.scalar_type() == torch::kInt32,
              "slot_indices must be int32");
  TORCH_CHECK(slot_indices.dim() == 1 &&
                  slot_indices.size(0) == logits.size(0),
              "slot_indices must have shape [B]");
  TORCH_CHECK(states.device() == logits.device() &&
                  slot_indices.device() == logits.device(),
              "states and slot_indices must share logits' CUDA device");
}

void check_penalties(const torch::Tensor& logits,
                     const torch::Tensor& penalties,
                     const torch::Tensor& states) {
  check_cuda_contiguous(penalties, "penalties");
  TORCH_CHECK(penalties.scalar_type() == torch::kFloat32,
              "penalties must be float32");
  TORCH_CHECK(penalties.dim() == 2 &&
                  penalties.size(0) == states.size(0) &&
                  penalties.size(1) == logits.size(1),
              "penalties must have shape [num_slots,V]");
  TORCH_CHECK(penalties.device() == logits.device(),
              "penalties must share logits' CUDA device");
}

void check_parameter(const torch::Tensor& logits,
                     const torch::Tensor& parameter, const char* name,
                     torch::ScalarType dtype) {
  check_cuda_contiguous(parameter, name);
  TORCH_CHECK(parameter.scalar_type() == dtype, name,
              " has an invalid dtype");
  TORCH_CHECK(parameter.dim() == 1 && parameter.size(0) == logits.size(0),
              name, " must have shape [B]");
  TORCH_CHECK(parameter.device() == logits.device(), name,
              " must share logits' CUDA device");
}

void check_finite(double value, const char* name) {
  TORCH_CHECK(std::isfinite(value), name, " must be finite");
}

torch::Tensor check_sampling_activity(
    const torch::Tensor& logits,
    int64_t sample_capacity,
    const std::optional<torch::Tensor>& num_active_samples) {
  if (!num_active_samples.has_value()) {
    TORCH_CHECK(
        sample_capacity == -1 || sample_capacity == logits.size(0),
        "sample_capacity must equal the logits row capacity");
    return torch::Tensor();
  }
  TORCH_CHECK(
      sample_capacity == logits.size(0),
      "sample_capacity must equal the logits row capacity");
  check_cuda_contiguous(*num_active_samples, "num_active_samples");
  TORCH_CHECK(
      num_active_samples->scalar_type() == torch::kInt32 &&
          num_active_samples->numel() == 1,
      "num_active_samples must be a one-element CUDA int32 tensor");
  TORCH_CHECK(
      num_active_samples->device() == logits.device(),
      "num_active_samples must share logits' CUDA device");
  return *num_active_samples;
}

}  // namespace

torch::Tensor setup_sampling_states(int64_t seed, int64_t num_slots) {
  TORCH_CHECK(num_slots > 0 && num_slots <= std::numeric_limits<int>::max(),
              "num_slots must be in [1, INT_MAX]");
  return setup_sampling_states_cuda(seed, num_slots);
}

torch::Tensor sampling_temperature_topk_topp_scalar(
    torch::Tensor logits, torch::Tensor states, torch::Tensor slot_indices,
    double temperature, int64_t top_k, double top_p,
    int64_t sample_capacity,
    std::optional<torch::Tensor> num_active_samples) {
  check_logits(logits);
  check_state_and_slots(logits, states, slot_indices);
  check_finite(temperature, "temperature");
  check_finite(top_p, "top_p");
  auto active = check_sampling_activity(
      logits, sample_capacity, num_active_samples);
  const c10::cuda::CUDAGuard device_guard(logits.device());
  return sampling_temperature_topk_topp_scalar_cuda(
      logits, states, slot_indices, temperature, top_k, top_p, active);
}

torch::Tensor sampling_temperature_topk_topp_per_request(
    torch::Tensor logits, torch::Tensor states, torch::Tensor slot_indices,
    torch::Tensor temperatures, torch::Tensor top_ks, torch::Tensor top_ps,
    int64_t sample_capacity,
    std::optional<torch::Tensor> num_active_samples) {
  check_logits(logits);
  check_state_and_slots(logits, states, slot_indices);
  check_parameter(logits, temperatures, "temperatures", torch::kFloat32);
  check_parameter(logits, top_ks, "top_ks", torch::kInt32);
  check_parameter(logits, top_ps, "top_ps", torch::kFloat32);
  auto active = check_sampling_activity(
      logits, sample_capacity, num_active_samples);
  const c10::cuda::CUDAGuard device_guard(logits.device());
  return sampling_temperature_topk_topp_per_request_cuda(
      logits, states, slot_indices, temperatures, top_ks, top_ps, active);
}

torch::Tensor sampling_six_parameter_scalar(
    torch::Tensor logits, torch::Tensor penalties, torch::Tensor states,
    torch::Tensor slot_indices, double presence_penalty,
    double frequency_penalty, double penalty_decay, double temperature,
    int64_t top_k, double top_p, int64_t sample_capacity,
    std::optional<torch::Tensor> num_active_samples) {
  check_logits(logits);
  check_state_and_slots(logits, states, slot_indices);
  check_penalties(logits, penalties, states);
  check_finite(presence_penalty, "presence_penalty");
  check_finite(frequency_penalty, "frequency_penalty");
  check_finite(penalty_decay, "penalty_decay");
  check_finite(temperature, "temperature");
  check_finite(top_p, "top_p");
  auto active = check_sampling_activity(
      logits, sample_capacity, num_active_samples);
  const c10::cuda::CUDAGuard device_guard(logits.device());
  return sampling_six_parameter_scalar_cuda(
      logits, penalties, states, slot_indices, presence_penalty,
      frequency_penalty, penalty_decay, temperature, top_k, top_p, active);
}

torch::Tensor sampling_six_parameter_per_request(
    torch::Tensor logits, torch::Tensor penalties, torch::Tensor states,
    torch::Tensor slot_indices, torch::Tensor presence_penalties,
    torch::Tensor frequency_penalties, torch::Tensor penalty_decays,
    torch::Tensor temperatures, torch::Tensor top_ks, torch::Tensor top_ps,
    int64_t sample_capacity,
    std::optional<torch::Tensor> num_active_samples) {
  check_logits(logits);
  check_state_and_slots(logits, states, slot_indices);
  check_penalties(logits, penalties, states);
  check_parameter(logits, presence_penalties, "presence_penalties",
                  torch::kFloat32);
  check_parameter(logits, frequency_penalties, "frequency_penalties",
                  torch::kFloat32);
  check_parameter(logits, penalty_decays, "penalty_decays", torch::kFloat32);
  check_parameter(logits, temperatures, "temperatures", torch::kFloat32);
  check_parameter(logits, top_ks, "top_ks", torch::kInt32);
  check_parameter(logits, top_ps, "top_ps", torch::kFloat32);
  auto active = check_sampling_activity(
      logits, sample_capacity, num_active_samples);
  const c10::cuda::CUDAGuard device_guard(logits.device());
  return sampling_six_parameter_per_request_cuda(
      logits, penalties, states, slot_indices, presence_penalties,
      frequency_penalties, penalty_decays, temperatures, top_ks, top_ps,
      active);
}

void register_sampling_bindings(py::module_& module) {
  module.def("setup_sampling_states", &setup_sampling_states, py::arg("seed"),
             py::arg("num_slots"));
  module.def("sampling_temperature_topk_topp_scalar",
             &sampling_temperature_topk_topp_scalar, py::arg("logits"),
             py::arg("states"), py::arg("slot_indices"),
             py::arg("temperature"), py::arg("top_k"), py::arg("top_p"),
             py::arg("sample_capacity") = -1,
             py::arg("num_active_samples") = py::none());
  module.def("sampling_temperature_topk_topp_per_request",
             &sampling_temperature_topk_topp_per_request,
             py::arg("logits"), py::arg("states"), py::arg("slot_indices"),
             py::arg("temperatures"), py::arg("top_ks"), py::arg("top_ps"),
             py::arg("sample_capacity") = -1,
             py::arg("num_active_samples") = py::none());
  module.def("sampling_six_parameter_scalar",
             &sampling_six_parameter_scalar, py::arg("logits"),
             py::arg("penalties"), py::arg("states"),
             py::arg("slot_indices"), py::arg("presence_penalty"),
             py::arg("frequency_penalty"), py::arg("penalty_decay"),
             py::arg("temperature"), py::arg("top_k"), py::arg("top_p"),
             py::arg("sample_capacity") = -1,
             py::arg("num_active_samples") = py::none());
  module.def("sampling_six_parameter_per_request",
             &sampling_six_parameter_per_request, py::arg("logits"),
             py::arg("penalties"), py::arg("states"),
             py::arg("slot_indices"), py::arg("presence_penalties"),
             py::arg("frequency_penalties"), py::arg("penalty_decays"),
             py::arg("temperatures"), py::arg("top_ks"), py::arg("top_ps"),
             py::arg("sample_capacity") = -1,
             py::arg("num_active_samples") = py::none());
}
