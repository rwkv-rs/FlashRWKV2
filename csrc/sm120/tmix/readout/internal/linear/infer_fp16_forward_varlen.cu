// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to BlinkDL/Albatross
// Albatross source revision: ee3308f6922e59f2166c7fac3c5a192340a2b48e.
// TMix Readout owns caller-specific dispatch and the public fusion contract.
// Shared SM120 CUDA primitives live in the unregistered native-private provider.

#include "../../../../internal/linear/backend.cuh"

at::Tensor tmix_readout_projection_dispatch_f16_cuda(
    at::Tensor x, at::Tensor weight_orig) {
  const int64_t c = x.size(-1);
  const int64_t rows = x.numel() / c;

  // Exact C=4096 winners admitted by the canonical Albatross caller.  The
  // shape guards are part of the policy; do not broaden these tables to other
  // model widths or callers.
  if (c == 4096) {
    if (weight_orig.size(0) == 4096 &&
        weight_orig.size(1) == 4096) {
      switch (rows) {
        case 24:
          return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 0, 0);
        case 32:
        case 96:
          return linear_f16_orig_cuda(x, weight_orig);
        case 48:
        case 64:
          return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 32, 0);
        case 192:
          return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 32, 2);
        default:
          break;
      }
    }
  }

  if (rows == 1) {
    return linear_orig_rows_exact_f16_cuda(
        x, weight_orig, 128, 2,
        c < 2048);
  }
  if (rows == 2) {
    return linear_orig_rows_exact_f16_cuda(x, weight_orig, 64, 2, true);
  }
  if (rows == 3) {
    if (c == 768) return linear_orig_rows_f16_cuda(x, weight_orig, 1, 2);
    if (c == 1024) return linear_orig_rows_f16_cuda(x, weight_orig, 2, 2);
    if (c == 2048) return linear_orig_rows_f16_cuda(x, weight_orig, 3, 4);
    if (c == 2560) return linear_orig_rows_f16_cuda(x, weight_orig, 3, 2);
    return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 0, 2);
  }
  if (rows == 4) {
    if (c <= 1024) return linear_orig_rows_f16_cuda(x, weight_orig, 2, 2);
    if (c == 2048 || c == 2560) {
      return linear_orig_rows_f16_cuda(x, weight_orig, 4, 2);
    }
    return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 0, 2);
  }

  // The remaining shape windows are the canonical caller-specific Lt policy.
  // It is intentionally explicit; an unlisted shape ends in the upstream
  // original-layout GEMM rather than a new local fallback implementation.

    if (c == 2560 && rows >= 17 && rows <= 20) {
      return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 0, 0);
    }
    if (c == 768) {
      if (rows >= 256 && rows < 384) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 128, 1);
      if (rows >= 96 && rows < 112) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 32, 3);
    }
    if (c == 1024) {
      if (rows >= 256 && rows < 384) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 128, 0);
      if (rows >= 96 && rows < 112) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 32, 6);
    }
    if (c == 2048) {
      if (rows >= 256 && rows < 384) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 32, 3);
      if (rows >= 192 && rows < 256) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 128, 0);
      if (rows >= 96 && rows < 112) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 32, 4);
    }
    if (c == 2560) {
      if (rows >= 256) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 0, 1);
      if (rows >= 160) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 0, 2);
      if (rows >= 128) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 128, 2);
      if (rows >= 112) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 128, 3);
      if (rows >= 96) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 32, 2);
      if (rows >= 72) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 128, 2);
      if (rows >= 5) return linear_f16_orig_cuda(x, weight_orig);
    }
    if (rows >= 1024) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 32, 4);
    if (rows >= 768) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 32, 0);
    if (rows >= 512) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 32, 1);
    if (rows >= 384) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 128, 2);
    if (rows >= 256) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 32, 4);
    if (rows >= 192) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 0, 0);
    if (rows >= 160) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 128, 1);
    if (rows >= 112) return linear_f16_orig_cuda(x, weight_orig);
    if (rows >= 96) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 0, 5);
    if (rows >= 72) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 32, 0);
    if (rows >= 48) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 32, 6);
    if (rows >= 32) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 0, 0);
    if (rows >= 24) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 0, 6);
    if (rows >= 12) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 0, 0);
    if (rows >= 5) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 0, 2);

  return linear_f16_orig_cuda(x, weight_orig);
}
