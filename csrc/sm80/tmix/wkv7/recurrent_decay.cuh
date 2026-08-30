// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the FlashRWKV2 project

#pragma once

#include <cuda_runtime.h>

namespace flashrwkv2::wkv7 {

constexpr float kNegativeExpHalfLog2E = -0.8750387749145276f;
constexpr float kNegativeLog2E = -1.4426950408889634f;
constexpr float kExpNegativeHalf = 0.6065306597126334f;
constexpr float kDitherUnit = 4.547473508864641e-13f;
constexpr uint32_t kDitherMultiplier = 2654435769u;

// The product boundary is raw RWKV-7 decay logits.  Keep the producer
// transform here so inference does not materialize a product log-decay tensor.
__device__ __forceinline__ float recurrent_retention(float decay_logits) {
  return exp2f(
      kNegativeExpHalfLog2E /
      (1.0f + exp2f(kNegativeLog2E * decay_logits)));
}

// This is d(log(retention))/d(decay_logits).  The recurrent backward
// recurrence first contracts through the retention value and then applies
// this logarithmic derivative, which is the train_temp raw-decay boundary.
__device__ __forceinline__ float recurrent_retention_log_derivative(
    float decay_logits) {
  const float sigmoid =
      1.0f / (1.0f + exp2f(kNegativeLog2E * decay_logits));
  return -kExpNegativeHalf * sigmoid * (1.0f - sigmoid);
}

// Exact Albatross FP16-state w_delta boundary.  The FP16 recurrent update
// applies `state + state * delta`; elapsed state supplies the phase term.
__device__ __forceinline__ float recurrent_fp16_dither(int phase) {
  const uint32_t bits =
      kDitherMultiplier * static_cast<uint32_t>(phase);
  return kDitherUnit * static_cast<float>(static_cast<int32_t>(bits));
}

__device__ __forceinline__ float recurrent_fp16_delta(
    float decay_logits, int phase) {
  return recurrent_retention(decay_logits) - 1.0f +
      recurrent_fp16_dither(phase);
}

}  // namespace flashrwkv2::wkv7
