// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Adapted from vllm-rwkv rwkv7_wkv_fp32_v2 at commit
// 6d683f9e49a2997e405c47edc147872c8609513b.

#include "../bindings.h"
#include "../validation.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

using flashrwkv2::validation::check_cuda_contiguous;
using flashrwkv2::validation::check_same_device;
using flashrwkv2::validation::prepare_recurrent_graph_metadata_cuda;
using flashrwkv2::validation::prepare_recurrent_metadata_cuda;

namespace {

int64_t check_recurrent_metadata_summary(
    const torch::Tensor& validation_summary, int64_t total_tokens) {
  auto summary_cpu = validation_summary.to(torch::kCPU);
  const auto* summary = summary_cpu.data_ptr<int32_t>();
  TORCH_CHECK(summary[2] == 0, "query_start_loc must start at 0");
  TORCH_CHECK(
      summary[3] == total_tokens,
      "the final query_start_loc offset must equal total_tokens");
  TORCH_CHECK(
      (summary[0] & flashrwkv2::validation::kInvalidSequenceRange) == 0,
      "query_start_loc must be strictly increasing with non-empty sequences");
  TORCH_CHECK(
      (summary[0] & flashrwkv2::validation::kInvalidStateSlot) == 0,
      "state_indices entries must be within the state pool");
  TORCH_CHECK(
      (summary[0] & flashrwkv2::validation::kDuplicateStateSlot) == 0,
      "state_indices must be unique within one call");
  return summary[1];
}

std::optional<uint32_t> tensor_version(const torch::Tensor& tensor) {
  if (tensor.unsafeGetTensorImpl()->is_inference()) {
    return std::nullopt;
  }
  return tensor.unsafeGetTensorImpl()->version_counter().current_version();
}

class RecurrentMetadataTicket final {
 public:
  RecurrentMetadataTicket(
      torch::Tensor query_start_loc,
      torch::Tensor state_indices,
      torch::Tensor query_start_loc_snapshot,
      torch::Tensor state_indices_snapshot,
      torch::Tensor status,
      torch::Tensor workspace,
      torch::Tensor num_active_tokens,
      torch::Tensor num_active_sequences,
      int64_t total_tokens,
      int64_t state_pool_size,
      int64_t max_seqlen,
      bool graph_mode,
      cudaStream_t stream)
      : query_start_loc_(std::move(query_start_loc)),
        state_indices_(std::move(state_indices)),
        query_start_loc_snapshot_(std::move(query_start_loc_snapshot)),
        state_indices_snapshot_(std::move(state_indices_snapshot)),
        status_(std::move(status)),
        workspace_(std::move(workspace)),
        num_active_tokens_(std::move(num_active_tokens)),
        num_active_sequences_(std::move(num_active_sequences)),
        query_start_loc_version_(tensor_version(query_start_loc_)),
        state_indices_version_(tensor_version(state_indices_)),
        query_start_loc_data_(query_start_loc_.data_ptr()),
        state_indices_data_(state_indices_.data_ptr()),
        query_start_loc_sizes_(query_start_loc_.sizes().vec()),
        state_indices_sizes_(state_indices_.sizes().vec()),
        query_start_loc_strides_(query_start_loc_.strides().vec()),
        state_indices_strides_(state_indices_.strides().vec()),
        total_tokens_(total_tokens),
        state_pool_size_(state_pool_size),
        max_seqlen_(max_seqlen),
        graph_mode_(graph_mode),
        device_(query_start_loc_.device()),
        stream_(stream) {}

  void check_compatible(
      const torch::Tensor& query_start_loc,
      const torch::Tensor& state_indices,
      int64_t total_tokens,
      int64_t state_pool_size,
      int64_t max_seqlen) const {
    TORCH_CHECK(
        query_start_loc.is_same(query_start_loc_),
        "validated_metadata query_start_loc identity mismatch");
    TORCH_CHECK(
        state_indices.is_same(state_indices_),
        "validated_metadata state_indices identity mismatch");
    TORCH_CHECK(
        query_start_loc.data_ptr() == query_start_loc_data_ &&
            state_indices.data_ptr() == state_indices_data_,
        "validated_metadata metadata data_ptr mismatch");
    TORCH_CHECK(
        query_start_loc.sizes().vec() == query_start_loc_sizes_ &&
            state_indices.sizes().vec() == state_indices_sizes_ &&
            query_start_loc.strides().vec() == query_start_loc_strides_ &&
            state_indices.strides().vec() == state_indices_strides_,
        "validated_metadata metadata shape or stride mismatch");
    if (!graph_mode_ && query_start_loc_version_.has_value()) {
      TORCH_CHECK(
          tensor_version(query_start_loc) == query_start_loc_version_,
          "validated_metadata query_start_loc version mismatch");
    }
    if (!graph_mode_ && state_indices_version_.has_value()) {
      TORCH_CHECK(
          tensor_version(state_indices) == state_indices_version_,
          "validated_metadata state_indices version mismatch");
    }
    TORCH_CHECK(
        query_start_loc.device() == device_ && state_indices.device() == device_,
        "validated_metadata device mismatch");
    TORCH_CHECK(
        total_tokens == total_tokens_,
        "validated_metadata total_tokens mismatch");
    TORCH_CHECK(
        state_pool_size == state_pool_size_,
        "validated_metadata state_pool_size mismatch");
    if (max_seqlen > 0) {
      TORCH_CHECK(
          max_seqlen == max_seqlen_,
          "validated_metadata max_seqlen mismatch");
    }
    TORCH_CHECK(
        at::cuda::getCurrentCUDAStream(device_.index()).stream() == stream_,
        "validated_metadata stream mismatch; prepare and consume the ticket "
        "on the same CUDA stream");
  }

  const torch::Tensor& query_start_loc_snapshot() const {
    return query_start_loc_snapshot_;
  }

  const torch::Tensor& state_indices_snapshot() const {
    return state_indices_snapshot_;
  }

  const torch::Tensor& status() const { return status_; }

  int64_t max_seqlen() const { return max_seqlen_; }

  bool graph_mode() const { return graph_mode_; }

  const torch::Tensor& num_active_tokens() const {
    return num_active_tokens_;
  }

  const torch::Tensor& num_active_sequences() const {
    return num_active_sequences_;
  }

 private:
  torch::Tensor query_start_loc_;
  torch::Tensor state_indices_;
  torch::Tensor query_start_loc_snapshot_;
  torch::Tensor state_indices_snapshot_;
  torch::Tensor status_;
  torch::Tensor workspace_;
  torch::Tensor num_active_tokens_;
  torch::Tensor num_active_sequences_;
  std::optional<uint32_t> query_start_loc_version_;
  std::optional<uint32_t> state_indices_version_;
  void* query_start_loc_data_;
  void* state_indices_data_;
  std::vector<int64_t> query_start_loc_sizes_;
  std::vector<int64_t> state_indices_sizes_;
  std::vector<int64_t> query_start_loc_strides_;
  std::vector<int64_t> state_indices_strides_;
  int64_t total_tokens_;
  int64_t state_pool_size_;
  int64_t max_seqlen_;
  bool graph_mode_;
  c10::Device device_;
  cudaStream_t stream_;
};

std::shared_ptr<RecurrentMetadataTicket> prepare_tmix_wkv7_recurrent_metadata(
    torch::Tensor query_start_loc,
    torch::Tensor state_indices,
    int64_t total_tokens,
    int64_t state_pool_size,
    int64_t max_seqlen) {
  check_cuda_contiguous(query_start_loc, "query_start_loc");
  check_cuda_contiguous(state_indices, "state_indices");
  check_same_device(query_start_loc, state_indices, "state_indices");
  TORCH_CHECK(
      query_start_loc.scalar_type() == torch::kInt32 &&
          state_indices.scalar_type() == torch::kInt32,
      "recurrent metadata must be int32");
  TORCH_CHECK(
      query_start_loc.dim() == 1 && state_indices.dim() == 1 &&
          state_indices.numel() > 0 &&
          query_start_loc.numel() == state_indices.numel() + 1,
      "query_start_loc must have shape [B+1] and state_indices shape [B]");
  TORCH_CHECK(
      state_indices.numel() <= 65535,
      "state_indices must contain at most 65535 sequences");
  TORCH_CHECK(
      total_tokens > 0 && total_tokens <= std::numeric_limits<int>::max(),
      "total_tokens must be positive and fit in int32");
  TORCH_CHECK(
      state_pool_size > 0 &&
          state_pool_size <= std::numeric_limits<int>::max(),
      "state_pool_size must be positive and fit in int32");
  const c10::cuda::CUDAGuard device_guard(query_start_loc.device());
  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(query_start_loc.device().index()).stream();
  auto summary = torch::empty({4}, query_start_loc.options());
  auto prepared = prepare_recurrent_metadata_cuda(
      query_start_loc, state_indices, total_tokens, state_pool_size, summary);
  const int64_t inferred_max_seqlen = check_recurrent_metadata_summary(summary, total_tokens);
  if (max_seqlen <= 0) {
    max_seqlen = inferred_max_seqlen;
  } else {
    TORCH_CHECK(
        max_seqlen == inferred_max_seqlen,
        "max_seqlen must equal the largest packed sequence length");
  }
  return std::make_shared<RecurrentMetadataTicket>(
      std::move(query_start_loc), std::move(state_indices),
      std::move(prepared.query_start_loc),
      std::move(prepared.state_indices), std::move(prepared.status),
      std::move(prepared.workspace), torch::Tensor(), torch::Tensor(),
      total_tokens, state_pool_size, max_seqlen, false, stream);
}

std::shared_ptr<RecurrentMetadataTicket> prepare_tmix_wkv7_recurrent_graph_metadata(
    torch::Tensor query_start_loc,
    torch::Tensor state_indices,
    torch::Tensor num_active_tokens,
    torch::Tensor num_active_sequences,
    int64_t token_capacity,
    int64_t sequence_capacity,
    int64_t state_pool_size,
    int64_t max_seqlen_capacity) {
  check_cuda_contiguous(query_start_loc, "query_start_loc");
  check_cuda_contiguous(state_indices, "state_indices");
  check_same_device(query_start_loc, state_indices, "state_indices");
  TORCH_CHECK(
      query_start_loc.scalar_type() == torch::kInt32 &&
          state_indices.scalar_type() == torch::kInt32,
      "recurrent metadata must be int32");
  TORCH_CHECK(
      sequence_capacity > 0 && sequence_capacity <= 65535 &&
          state_indices.dim() == 1 &&
          state_indices.numel() == sequence_capacity &&
          query_start_loc.dim() == 1 &&
          query_start_loc.numel() == sequence_capacity + 1,
      "graph metadata shapes must match sequence_capacity");
  TORCH_CHECK(
      token_capacity > 0 && token_capacity <= std::numeric_limits<int>::max(),
      "token_capacity must be positive and fit in int32");
  TORCH_CHECK(
      state_pool_size > 0 &&
          state_pool_size <= std::numeric_limits<int>::max(),
      "state_pool_size must be positive and fit in int32");
  TORCH_CHECK(
      max_seqlen_capacity > 0 &&
          max_seqlen_capacity <= token_capacity,
      "max_seqlen_capacity must be positive and not exceed token_capacity");

  const c10::cuda::CUDAGuard device_guard(query_start_loc.device());
  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(query_start_loc.device().index()).stream();
  auto prepared = prepare_recurrent_graph_metadata_cuda(
      query_start_loc, state_indices, num_active_tokens,
      num_active_sequences, token_capacity, sequence_capacity,
      state_pool_size, max_seqlen_capacity);
  return std::make_shared<RecurrentMetadataTicket>(
      std::move(query_start_loc), std::move(state_indices),
      std::move(prepared.query_start_loc),
      std::move(prepared.state_indices), std::move(prepared.status),
      std::move(prepared.workspace), std::move(num_active_tokens),
      std::move(num_active_sequences), token_capacity, state_pool_size,
      max_seqlen_capacity, true, stream);
}

}  // namespace

void register_recurrent_metadata_bindings(py::module_& module) {
  py::class_<
      RecurrentMetadataTicket,
      std::shared_ptr<RecurrentMetadataTicket>>(
      module, "_RecurrentMetadataTicket")
      .def(
          "_check_compatible",
          &RecurrentMetadataTicket::check_compatible,
          py::arg("query_start_loc"), py::arg("state_indices"),
          py::arg("total_tokens"), py::arg("state_pool_size"),
          py::arg("max_seqlen"))
      .def(
          "_query_start_loc_snapshot",
          [](const std::shared_ptr<RecurrentMetadataTicket>& ticket) {
            return ticket->query_start_loc_snapshot();
          })
      .def(
          "_state_indices_snapshot",
          [](const std::shared_ptr<RecurrentMetadataTicket>& ticket) {
            return ticket->state_indices_snapshot();
          })
      .def(
          "_status",
          [](const std::shared_ptr<RecurrentMetadataTicket>& ticket) {
            return ticket->status();
          })
      .def(
          "_active_status",
          [](const std::shared_ptr<RecurrentMetadataTicket>& ticket) {
            return ticket->status();
          })
      .def(
          "_max_seqlen",
          [](const std::shared_ptr<RecurrentMetadataTicket>& ticket) {
            return ticket->max_seqlen();
          })
      .def(
          "_is_graph",
          [](const std::shared_ptr<RecurrentMetadataTicket>& ticket) {
            return ticket->graph_mode();
          })
      .def(
          "_num_active_tokens",
          [](const std::shared_ptr<RecurrentMetadataTicket>& ticket) {
            return ticket->num_active_tokens();
          })
      .def(
          "_num_active_sequences",
          [](const std::shared_ptr<RecurrentMetadataTicket>& ticket) {
            return ticket->num_active_sequences();
          });
  module.def(
      "prepare_tmix_wkv7_recurrent_metadata", &prepare_tmix_wkv7_recurrent_metadata,
      "Prepare packed recurrent metadata for same-stream reuse",
      py::arg("query_start_loc"), py::arg("state_indices"),
      py::arg("total_tokens"), py::arg("state_pool_size"),
      py::arg("max_seqlen") = -1);
  module.def(
      "prepare_tmix_wkv7_recurrent_graph_metadata", &prepare_tmix_wkv7_recurrent_graph_metadata,
      "Prepare live packed recurrent metadata for warmup or CUDA Graph capture",
      py::arg("query_start_loc"), py::arg("state_indices"),
      py::arg("num_active_tokens"), py::arg("num_active_sequences"),
      py::arg("token_capacity"), py::arg("sequence_capacity"),
      py::arg("state_pool_size"), py::arg("max_seqlen_capacity"));
}
