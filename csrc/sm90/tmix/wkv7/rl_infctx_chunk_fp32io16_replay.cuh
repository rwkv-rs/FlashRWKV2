// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright contributors to the FlashRWKV2 project
// Canonical module owner: tmix/wkv7; RL/Infctx is the workload.
// Mechanically migrated from the RL/Infctx chunk replay source retained in
// the pre-refactor tree; raw-decay-only adaptation for this public contract.

#pragma once

#include <cuda_runtime.h>
#include <torch/extension.h>

void launch_rl_infctx_chunk_replay_fp32_from_decay_logits(
    int num_chunks,
    int num_heads,
    const torch::Tensor& chunk_token_starts,
    const torch::Tensor& chunk_token_ends,
    const torch::Tensor& boundary,
    const torch::Tensor& r,
    const torch::Tensor& decay_logits,
    const torch::Tensor& decay_bias,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor& output,
    torch::Tensor* state_dot_a,
    float scale,
    cudaStream_t stream);

void rl_infctx_chunk_fp32io16_backward_replay_cuda(
    torch::Tensor chunk_token_starts,
    torch::Tensor chunk_token_ends,
    torch::Tensor boundary,
    torch::Tensor r,
    torch::Tensor decay_logits,
    torch::Tensor decay_bias,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor output,
    torch::Tensor state_dot_a,
    double scale);
