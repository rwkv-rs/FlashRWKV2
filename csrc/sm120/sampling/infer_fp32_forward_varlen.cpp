// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright contributors to Rapid-Sampling
// Adapted from Rapid-Sampling revision e0297f7830c3fa581d49ddddddba32f35ea7f733
// and rwkv-rs/vllm-rwkv revision fd440426689f10e240b5761e1a7c82e4c37deb8d.

#include "validation.h"


#include <cmath>
#include <limits>
#include <optional>

int64_t sampling_state_size_cuda();
torch::stable::Tensor setup_sampling_states_cuda(
    torch::stable::Tensor anchor, int64_t seed, int64_t num_slots);
torch::stable::Tensor sampling_temperature_topk_topp_scalar_cuda(
    torch::stable::Tensor logits, torch::stable::Tensor states, torch::stable::Tensor slot_indices,
    double temperature, int64_t top_k, double top_p,
    torch::stable::Tensor num_active_samples);
torch::stable::Tensor sampling_temperature_topk_topp_per_request_cuda(
    torch::stable::Tensor logits, torch::stable::Tensor states, torch::stable::Tensor slot_indices,
    torch::stable::Tensor temperatures, torch::stable::Tensor top_ks, torch::stable::Tensor top_ps,
    torch::stable::Tensor num_active_samples);
torch::stable::Tensor sampling_six_parameter_scalar_cuda(
    torch::stable::Tensor logits, torch::stable::Tensor penalties, torch::stable::Tensor states,
    torch::stable::Tensor slot_indices, double presence_penalty,
    double frequency_penalty, double penalty_decay, double temperature,
    int64_t top_k, double top_p, torch::stable::Tensor num_active_samples);
torch::stable::Tensor sampling_six_parameter_per_request_cuda(
    torch::stable::Tensor logits, torch::stable::Tensor penalties, torch::stable::Tensor states,
    torch::stable::Tensor slot_indices, torch::stable::Tensor presence_penalties,
    torch::stable::Tensor frequency_penalties, torch::stable::Tensor penalty_decays,
    torch::stable::Tensor temperatures, torch::stable::Tensor top_ks, torch::stable::Tensor top_ps,
    torch::stable::Tensor num_active_samples);

namespace {

void check_cuda_contiguous(const torch::stable::Tensor& tensor, const char* name) {
  STD_TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
  STD_TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_logits(const torch::stable::Tensor& logits) {
  check_cuda_contiguous(logits, "logits");
  STD_TORCH_CHECK(logits.scalar_type() == torch::headeronly::ScalarType::Float,
              "logits must be float32");
  STD_TORCH_CHECK(logits.dim() == 2 && logits.size(0) > 0,
              "logits must have compact request shape [B,V]");
  const int64_t vocab_size = logits.size(1);
  STD_TORCH_CHECK(vocab_size > 0 && vocab_size <= 1048576 && vocab_size % 4 == 0,
              "vocabulary size must be positive, divisible by 4, and no larger "
              "than 1048576");
  STD_TORCH_CHECK(logits.size(0) <= std::numeric_limits<int>::max() &&
                  vocab_size <= std::numeric_limits<int>::max(),
              "sampling dimensions must fit int32");
}

void check_state_and_slots(const torch::stable::Tensor& logits,
                           const torch::stable::Tensor& states,
                           const torch::stable::Tensor& slot_indices) {
  check_cuda_contiguous(states, "states");
  STD_TORCH_CHECK(states.scalar_type() == torch::headeronly::ScalarType::Char,
              "states must be int8 sampling state storage");
  STD_TORCH_CHECK(states.dim() == 2 && states.size(0) > 0 &&
                  states.size(1) == sampling_state_size_cuda(),
              "states must come from setup_sampling_states");
  check_cuda_contiguous(slot_indices, "slot_indices");
  STD_TORCH_CHECK(slot_indices.scalar_type() == torch::headeronly::ScalarType::Int,
              "slot_indices must be int32");
  STD_TORCH_CHECK(slot_indices.dim() == 1 &&
                  slot_indices.size(0) == logits.size(0),
              "slot_indices must have shape [B]");
  STD_TORCH_CHECK(states.device() == logits.device() &&
                  slot_indices.device() == logits.device(),
              "states and slot_indices must share logits' CUDA device");
}

void check_penalties(const torch::stable::Tensor& logits,
                     const torch::stable::Tensor& penalties,
                     const torch::stable::Tensor& states) {
  check_cuda_contiguous(penalties, "penalties");
  STD_TORCH_CHECK(penalties.scalar_type() == torch::headeronly::ScalarType::Float,
              "penalties must be float32");
  STD_TORCH_CHECK(penalties.dim() == 2 &&
                  penalties.size(0) == states.size(0) &&
                  penalties.size(1) == logits.size(1),
              "penalties must have shape [num_slots,V]");
  STD_TORCH_CHECK(penalties.device() == logits.device(),
              "penalties must share logits' CUDA device");
}

void check_parameter(const torch::stable::Tensor& logits,
                     const torch::stable::Tensor& parameter, const char* name,
                     torch::headeronly::ScalarType dtype) {
  check_cuda_contiguous(parameter, name);
  STD_TORCH_CHECK(parameter.scalar_type() == dtype, name,
              " has an invalid dtype");
  STD_TORCH_CHECK(parameter.dim() == 1 && parameter.size(0) == logits.size(0),
              name, " must have shape [B]");
  STD_TORCH_CHECK(parameter.device() == logits.device(), name,
              " must share logits' CUDA device");
}

void check_finite(double value, const char* name) {
  STD_TORCH_CHECK(std::isfinite(value), name, " must be finite");
}

torch::stable::Tensor check_sampling_activity(
    const torch::stable::Tensor& logits,
    int64_t sample_capacity,
    const std::optional<torch::stable::Tensor>& num_active_samples) {
  if (!num_active_samples.has_value()) {
    STD_TORCH_CHECK(
        sample_capacity == -1 || sample_capacity == logits.size(0),
        "sample_capacity must equal the logits row capacity");
    return torch::stable::Tensor();
  }
  STD_TORCH_CHECK(
      sample_capacity == logits.size(0),
      "sample_capacity must equal the logits row capacity");
  check_cuda_contiguous(*num_active_samples, "num_active_samples");
  STD_TORCH_CHECK(
      num_active_samples->scalar_type() == torch::headeronly::ScalarType::Int &&
          num_active_samples->numel() == 1,
      "num_active_samples must be a one-element CUDA int32 tensor");
  STD_TORCH_CHECK(
      num_active_samples->device() == logits.device(),
      "num_active_samples must share logits' CUDA device");
  return *num_active_samples;
}

}  // namespace

torch::stable::Tensor setup_sampling_states(
    torch::stable::Tensor anchor, int64_t seed, int64_t num_slots) {
  check_cuda_contiguous(anchor, "anchor");
  STD_TORCH_CHECK(num_slots > 0 && num_slots <= std::numeric_limits<int>::max(),
              "num_slots must be in [1, INT_MAX]");
  return setup_sampling_states_cuda(std::move(anchor), seed, num_slots);
}

torch::stable::Tensor sampling_temperature_topk_topp_scalar(
    torch::stable::Tensor logits, torch::stable::Tensor states, torch::stable::Tensor slot_indices,
    double temperature, int64_t top_k, double top_p,
    int64_t sample_capacity,
    std::optional<torch::stable::Tensor> num_active_samples) {
  check_logits(logits);
  check_state_and_slots(logits, states, slot_indices);
  check_finite(temperature, "temperature");
  check_finite(top_p, "top_p");
  auto active = check_sampling_activity(
      logits, sample_capacity, num_active_samples);
  const torch::stable::accelerator::DeviceGuard device_guard(logits.device().index());
  return sampling_temperature_topk_topp_scalar_cuda(
      logits, states, slot_indices, temperature, top_k, top_p, active);
}

torch::stable::Tensor sampling_temperature_topk_topp_per_request(
    torch::stable::Tensor logits, torch::stable::Tensor states, torch::stable::Tensor slot_indices,
    torch::stable::Tensor temperatures, torch::stable::Tensor top_ks, torch::stable::Tensor top_ps,
    int64_t sample_capacity,
    std::optional<torch::stable::Tensor> num_active_samples) {
  check_logits(logits);
  check_state_and_slots(logits, states, slot_indices);
  check_parameter(logits, temperatures, "temperatures", torch::headeronly::ScalarType::Float);
  check_parameter(logits, top_ks, "top_ks", torch::headeronly::ScalarType::Int);
  check_parameter(logits, top_ps, "top_ps", torch::headeronly::ScalarType::Float);
  auto active = check_sampling_activity(
      logits, sample_capacity, num_active_samples);
  const torch::stable::accelerator::DeviceGuard device_guard(logits.device().index());
  return sampling_temperature_topk_topp_per_request_cuda(
      logits, states, slot_indices, temperatures, top_ks, top_ps, active);
}

torch::stable::Tensor sampling_six_parameter_scalar(
    torch::stable::Tensor logits, torch::stable::Tensor penalties, torch::stable::Tensor states,
    torch::stable::Tensor slot_indices, double presence_penalty,
    double frequency_penalty, double penalty_decay, double temperature,
    int64_t top_k, double top_p, int64_t sample_capacity,
    std::optional<torch::stable::Tensor> num_active_samples) {
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
  const torch::stable::accelerator::DeviceGuard device_guard(logits.device().index());
  return sampling_six_parameter_scalar_cuda(
      logits, penalties, states, slot_indices, presence_penalty,
      frequency_penalty, penalty_decay, temperature, top_k, top_p, active);
}

torch::stable::Tensor sampling_six_parameter_per_request(
    torch::stable::Tensor logits, torch::stable::Tensor penalties, torch::stable::Tensor states,
    torch::stable::Tensor slot_indices, torch::stable::Tensor presence_penalties,
    torch::stable::Tensor frequency_penalties, torch::stable::Tensor penalty_decays,
    torch::stable::Tensor temperatures, torch::stable::Tensor top_ks, torch::stable::Tensor top_ps,
    int64_t sample_capacity,
    std::optional<torch::stable::Tensor> num_active_samples) {
  check_logits(logits);
  check_state_and_slots(logits, states, slot_indices);
  check_penalties(logits, penalties, states);
  check_parameter(logits, presence_penalties, "presence_penalties",
                  torch::headeronly::ScalarType::Float);
  check_parameter(logits, frequency_penalties, "frequency_penalties",
                  torch::headeronly::ScalarType::Float);
  check_parameter(logits, penalty_decays, "penalty_decays", torch::headeronly::ScalarType::Float);
  check_parameter(logits, temperatures, "temperatures", torch::headeronly::ScalarType::Float);
  check_parameter(logits, top_ks, "top_ks", torch::headeronly::ScalarType::Int);
  check_parameter(logits, top_ps, "top_ps", torch::headeronly::ScalarType::Float);
  auto active = check_sampling_activity(
      logits, sample_capacity, num_active_samples);
  const torch::stable::accelerator::DeviceGuard device_guard(logits.device().index());
  return sampling_six_parameter_per_request_cuda(
      logits, penalties, states, slot_indices, presence_penalties,
      frequency_penalties, penalty_decays, temperatures, top_ks, top_ps,
      active);
}

STABLE_TORCH_LIBRARY_FRAGMENT(flashrwkv2, module) {
  module.def("setup_sampling_states(Tensor anchor, int seed, int num_slots) -> Tensor");
  module.def("sampling_temperature_topk_topp_scalar_forward_varlen(Tensor logits, Tensor(a!) states, Tensor slot_indices, float temperature, int top_k, float top_p, int sample_capacity, Tensor? num_active_samples) -> Tensor");
  module.def("sampling_temperature_topk_topp_per_request_forward_varlen(Tensor logits, Tensor(a!) states, Tensor slot_indices, Tensor temperatures, Tensor top_ks, Tensor top_ps, int sample_capacity, Tensor? num_active_samples) -> Tensor");
  module.def("sampling_six_parameter_scalar_forward_varlen(Tensor logits, Tensor(a!) penalties, Tensor(b!) states, Tensor slot_indices, float presence_penalty, float frequency_penalty, float penalty_decay, float temperature, int top_k, float top_p, int sample_capacity, Tensor? num_active_samples) -> Tensor");
  module.def("sampling_six_parameter_per_request_forward_varlen(Tensor logits, Tensor(a!) penalties, Tensor(b!) states, Tensor slot_indices, Tensor presence_penalties, Tensor frequency_penalties, Tensor penalty_decays, Tensor temperatures, Tensor top_ks, Tensor top_ps, int sample_capacity, Tensor? num_active_samples) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(flashrwkv2, CUDA, module) {
  module.impl("setup_sampling_states", TORCH_BOX(&setup_sampling_states));
  module.impl("sampling_temperature_topk_topp_scalar_forward_varlen", TORCH_BOX(&sampling_temperature_topk_topp_scalar));
  module.impl("sampling_temperature_topk_topp_per_request_forward_varlen", TORCH_BOX(&sampling_temperature_topk_topp_per_request));
  module.impl("sampling_six_parameter_scalar_forward_varlen", TORCH_BOX(&sampling_six_parameter_scalar));
  module.impl("sampling_six_parameter_per_request_forward_varlen", TORCH_BOX(&sampling_six_parameter_per_request));
}
