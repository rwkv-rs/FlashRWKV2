// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright contributors to the FlashRWKV2 project
// Canonical module owner: tmix/wkv7; RL/Infctx is the workload.
// Mechanically migrated from the RL/Infctx chunk replay source retained in
// the pre-refactor tree; raw-decay-only adaptation for this public contract.

#pragma once

#include <cuda_runtime.h>
#include "validation.h"

void launch_rl_infctx_chunk_replay_fp32_from_decay_logits(
    int num_chunks,
    int num_heads,
    const torch::stable::Tensor& chunk_token_starts,
    const torch::stable::Tensor& chunk_token_ends,
    const torch::stable::Tensor& boundary,
    const torch::stable::Tensor& r,
    const torch::stable::Tensor& decay_logits,
    const torch::stable::Tensor& decay_bias,
    const torch::stable::Tensor& k,
    const torch::stable::Tensor& v,
    const torch::stable::Tensor& a,
    const torch::stable::Tensor& b,
    torch::stable::Tensor& output,
    torch::stable::Tensor* state_dot_a,
    float scale,
    cudaStream_t stream);

void rl_infctx_tmix_wkv7_chunk_fp32io16_backward_replay_cuda(
    torch::stable::Tensor chunk_token_starts,
    torch::stable::Tensor chunk_token_ends,
    torch::stable::Tensor boundary,
    torch::stable::Tensor r,
    torch::stable::Tensor decay_logits,
    torch::stable::Tensor decay_bias,
    torch::stable::Tensor k,
    torch::stable::Tensor v,
    torch::stable::Tensor a,
    torch::stable::Tensor b,
    torch::stable::Tensor output,
    torch::stable::Tensor state_dot_a,
    double scale);
