// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to BlinkDL/Albatross
// Upstream repository: https://github.com/BlinkDL/Albatross
// Albatross source revision: ee3308f6922e59f2166c7fac3c5a192340a2b48e
// Original path: faster3a_2607/cuda/rwkv7_v3a_ops.cu
// Mechanical migration boundary: the head caller reuses the exact
// Albatross original-layout linear, res and LN bodies owned
// by the private Linear provider and post_norm. Packed-row selection is local
// varlen adaptation; no generic GEMM or LN body is implemented here.

#include "../../internal/linear/backend.cuh"

#include <ATen/ATen.h>
at::Tensor head_linear_dispatch_f16_cuda(
    at::Tensor x, at::Tensor weight_orig) {
  const int64_t c = x.size(-1);
  const int64_t rows = x.numel() / c;

  // Exact C=4096 winners admitted by the canonical Albatross caller.  The
  // shape guards are part of the policy; do not broaden these tables to other
  // model widths or callers.
  if (rows == 1) {
    return linear_orig_rows_exact_f16_cuda(
        x, weight_orig, 128, 2, true);
  }
  if (rows == 2) {
    if (c == 2560) {
      return linear_orig_rows_exact_f16_cuda(x, weight_orig, 128, 2, false);
    }
    return linear_orig_rows_exact_f16_cuda(x, weight_orig, 64, 2, true);
  }
  if (rows == 3) {
    if (c <= 2560) {
      return linear_f16_orig_cuda(x, weight_orig);
    }
    return linear_orig_rows_f16_cuda(x, weight_orig, 3, 2);
  }
  // The remaining shape windows are the canonical caller-specific Lt policy.
  // It is intentionally explicit; an unlisted shape ends in the upstream
  // original-layout GEMM rather than a new local fallback implementation.

  if (c == 768) {
    if (rows >= 192 && rows < 256) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 128, 3);
    if (rows >= 96 && rows < 160) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 0, 1);
  }
  if (c == 1024) {
    if (rows >= 256 && rows < 384) return linear_f16_orig_cuda(x, weight_orig);
    if (rows >= 192 && rows < 256) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 0, 2);
    if (rows >= 96 && rows < 160) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 32, 1);
  }
  if (c == 2048) {
    if (rows >= 256 && rows < 384) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 32, 0);
    if (rows >= 192 && rows < 256) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 32, 6);
    if (rows >= 128 && rows < 160) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 0, 1);
    if (rows >= 96 && rows < 112) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 0, 0);
  }
  if (c == 2560) {
    if (rows >= 256) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 32, 0);
    if (rows >= 192) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 0, 5);
    if (rows >= 160) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 32, 5);
    if (rows >= 128) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 0, 1);
    if (rows >= 96) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 32, 0);
    if (rows >= 80) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 0, 0);
    if (rows >= 72) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 32, 1);
  }
  if (rows >= 1024) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 128, 0);
  if (rows >= 512) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 0, 2);
  if (rows >= 384) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 128, 2);
  if (rows >= 256) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 0, 1);
  if (rows >= 192) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 128, 0);
  if (rows >= 160) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 32, 0);
  if (rows >= 128) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 128, 0);
  if (rows >= 112) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 32, 0);
  if (rows >= 96) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 32, 1);
  if (rows >= 80) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 32, 2);
  if (rows >= 72) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 128, 2);

  return linear_f16_orig_cuda(x, weight_orig);
}

at::Tensor head_linear_all_forward_varlen_cuda(
    at::Tensor x, at::Tensor weight) {
  // Mechanical copy of HEAD_ALL_LOGITS_GEMM_4096 from the canonical
  // Albatross caller.  The table selects the existing original-layout Lt
  // body; an unlisted row count follows the existing head caller dispatch.
  if (x.size(1) == 4096) {
    switch (x.size(0)) {
      case 24:
        return linear_f16_orig_lt_cfg_cuda(x, weight, 0, 0);
      case 32:
        return linear_f16_orig_lt_cfg_cuda(x, weight, 0, 0);
      case 160:
        return linear_f16_orig_lt_cfg_cuda(x, weight, 0, 7);
      case 192:
        return linear_f16_orig_lt_cfg_cuda(x, weight, 0, 5);
      default:
        break;
    }
  }
  return head_linear_dispatch_f16_cuda(x, weight);
}

at::Tensor head_linear_last_forward_varlen_cuda(
    at::Tensor x, at::Tensor weight, int64_t tokens_count) {
  // Mechanical copy of HEAD_LAST_LOGITS_GEMM_4096.  This is deliberately
  // keyed by (rows, tokens_count): the last-logits GEMM has B rows even when
  // the preceding packed request contained B*T tokens.
  if (x.size(1) == 4096 && weight.size(0) == 65536 &&
      weight.size(1) == 4096) {
    const int64_t rows = x.size(0);
    if ((rows == 24 || rows == 32) &&
        (tokens_count == 1 || tokens_count == 2 || tokens_count == 4 ||
         tokens_count == 8 || (rows == 32 && tokens_count == 16))) {
      return linear_f16_orig_lt_cfg_cuda(x, weight, 0, 0);
    }
    if (rows == 160 &&
        (tokens_count == 1 || tokens_count == 2 || tokens_count == 4 ||
         tokens_count == 32)) {
      return linear_f16_orig_lt_cfg_cuda(x, weight, 0, 7);
    }
    if (rows == 192 &&
        (tokens_count == 1 || tokens_count == 2 || tokens_count == 4 ||
         tokens_count == 8 || tokens_count == 16 || tokens_count == 32)) {
      return linear_f16_orig_lt_cfg_cuda(x, weight, 0, 5);
    }
  }
  return head_linear_dispatch_f16_cuda(x, weight);
}
