// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to BlinkDL/Albatross
// SPDX-FileCopyrightText: Copyright contributors to the FlashRWKV2 project
// Native-private dense projection dispatch owned by the complete CMix operator.
// Albatross source revision: ee3308f6922e59f2166c7fac3c5a192340a2b48e.

#include "../../../internal/linear/backend.cuh"

#include <ATen/ATen.h>

at::Tensor cmix_linear_ffn_key_dispatch_f16_cuda(
    at::Tensor x, at::Tensor weight_orig) {
  const int64_t channels = x.size(-1);
  const int64_t rows = x.numel() / channels;
  if (FLASHRWKV_TARGET_SM != 120) {
    return linear_f16_orig_cuda(x, weight_orig);
  }

  if (channels == 4096 && weight_orig.size(0) == 16384 &&
      weight_orig.size(1) == 4096) {
    switch (rows) {
      case 24:
      case 32:
        return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 0, 2);
      case 128:
        return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 32, 3);
      case 192:
        return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 0, 3);
      case 384:
        return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 0, 2);
      default:
        break;
    }
  }

  if (rows == 1) {
    if (channels == 2560) {
      return linear_orig_rows_exact_f16_cuda(
          x, weight_orig, 128, 2, true);
    }
    return linear_orig_rows_exact_f16_cuda(
        x, weight_orig, 128, 2, channels <= 1024);
  }
  if (rows == 2) {
    if (channels == 2560) {
      return linear_orig_rows_exact_f16_cuda(
          x, weight_orig, 128, 2, false);
    }
    if (channels < 4096) {
      return linear_orig_rows_exact_f16_cuda(
          x, weight_orig, 64, 2, true);
    }
    return linear_orig_rows_exact_f16_cuda(
        x, weight_orig, 128, 2, false);
  }
  if (rows == 3) {
    if (channels <= 1024) {
      return linear_orig_rows_cfg_f16_cuda(
          x, weight_orig, 64, 3, 4);
    }
    if (channels == 2048 || channels == 2560) {
      return linear_f16_orig_cuda(x, weight_orig);
    }
    return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 0, 0);
  }
  if (rows == 4) {
    if (channels <= 1024) {
      return linear_orig_rows_cfg_f16_cuda(
          x, weight_orig, 64, 2, 4);
    }
    if (channels == 2048 || channels == 2560) {
      return linear_f16_orig_cuda(x, weight_orig);
    }
    return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 0, 0);
  }

  if (channels == 2560 && rows >= 17 && rows <= 20) {
    return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 0, 0);
  }
  if (channels == 768 &&
      ((rows >= 256 && rows < 384) || (rows >= 96 && rows < 112))) {
    return linear_f16_orig_cuda(x, weight_orig);
  }
  if (channels == 1024) {
    if (rows >= 256 && rows < 384)
      return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 32, 2);
    if (rows >= 192 && rows < 256)
      return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 0, 0);
    if (rows >= 96 && rows < 160)
      return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 32, 2);
  }
  if (channels == 2048 && rows >= 128 && rows < 160) {
    return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 0, 3);
  }
  if (channels == 2560) {
    if (rows >= 192) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 32, 5);
    if (rows >= 160) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 0, 4);
    if (rows >= 128) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 32, 5);
    if (rows >= 112) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 128, 4);
    if (rows >= 96) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 128, 4);
    if (rows >= 80) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 0, 3);
    if (rows >= 72) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 32, 4);
    if (rows >= 3) return linear_f16_orig_cuda(x, weight_orig);
  }
  if (rows >= 1024) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 0, 0);
  if (rows >= 768) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 32, 1);
  if (rows >= 512) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 128, 3);
  if (rows >= 384) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 32, 0);
  if (rows >= 256) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 128, 4);
  if (rows >= 192) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 0, 1);
  if (rows >= 160) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 0, 2);
  if (rows >= 128) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 32, 0);
  if (rows >= 112) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 32, 3);
  if (rows >= 96) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 32, 1);
  if (rows >= 72) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 128, 1);
  if (rows >= 48) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 0, 1);
  if (rows >= 12) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 0, 0);
  if (rows == 5 || rows == 6)
    return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 0, 1);
  return linear_f16_orig_cuda(x, weight_orig);
}

at::Tensor cmix_linear_ffn_down_forward_varlen_cuda(
    at::Tensor x, at::Tensor weight) {
  const int64_t rows = x.size(0);
  const bool canonical_4096_shape =
      x.size(1) == 16384 && weight.size(0) == 16384 &&
      weight.size(1) == 4096;
  if (FLASHRWKV_TARGET_SM == 120 && canonical_4096_shape) {
    if (rows == 48) {
      return linear_f16_lt_cfg_cuda(x, weight, 32, 1);
    }
    if (rows == 256) {
      return linear_f16_lt_cfg_cuda(x, weight, 32, 5);
    }
  }
  return internal_linear_transposed_f16_cuda(x, weight);
}
