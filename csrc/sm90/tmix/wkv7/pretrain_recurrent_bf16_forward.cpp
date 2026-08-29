// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to BlinkDL/RWKV-LM
// Canonical source: RWKV-LM RWKV-v7/train_temp/cuda/rwkv7_clampw_v3.cpp
// Source revision: 952102498e9ed367ea0a59ee64106916d474d30f.
// Local adaptation: runtime D=64/128/256 dispatch. D64 keeps clampw-v3,
// D128 follows rwkv7_clampw128_v2, and D256 uses a local warp-tiled extension.

#include "validation.h"

#ifdef _FP32_
    using bf = float;
#else
    #include <cuda_bf16.h>
    using bf = __nv_bfloat16;
#endif

void cuda_forward_v3(int B, int T, int H, bf*r, bf*w, bf*k, bf*v, bf*a, bf*b, bf*y, float*s, float*sa);
void cuda_forward_split(int B, int T, int H, int N, bf*r, bf*w, bf*k, bf*v, bf*a, bf*b, bf*y, float*s, float*sa);

void forward(torch::stable::Tensor &r, torch::stable::Tensor &w, torch::stable::Tensor &k, torch::stable::Tensor &v, torch::stable::Tensor &a, torch::stable::Tensor &b, torch::stable::Tensor &y, torch::stable::Tensor &s, torch::stable::Tensor &sa, int64_t head_size) {
    STD_TORCH_CHECK(head_size == 64 || head_size == 128 || head_size == 256,
                "head_size must be one of 64, 128, or 256");
    STD_TORCH_CHECK(r.size(3) == head_size, "input head dimension must equal head_size");
    int B = r.sizes()[0], T = r.sizes()[1], H = r.sizes()[2];
    if (head_size == 64) cuda_forward_v3(B, T, H, (bf*)r.mutable_data_ptr(), (bf*)w.mutable_data_ptr(), (bf*)k.mutable_data_ptr(), (bf*)v.mutable_data_ptr(), (bf*)a.mutable_data_ptr(), (bf*)b.mutable_data_ptr(), (bf*)y.mutable_data_ptr(), (float*)s.mutable_data_ptr(), (float*)sa.mutable_data_ptr());
    else cuda_forward_split(B, T, H, head_size, (bf*)r.mutable_data_ptr(), (bf*)w.mutable_data_ptr(), (bf*)k.mutable_data_ptr(), (bf*)v.mutable_data_ptr(), (bf*)a.mutable_data_ptr(), (bf*)b.mutable_data_ptr(), (bf*)y.mutable_data_ptr(), (float*)s.mutable_data_ptr(), (float*)sa.mutable_data_ptr());
}

void cuda_backward_v3(int B, int T, int H, bf*r, bf*w, bf*k, bf*v, bf*a, bf*b, bf*dy, float*s, float*sa, bf*dr, bf*dw, bf*dk, bf*dv, bf*da, bf*db);
void cuda_backward_split(int B, int T, int H, int N, bf*r, bf*w, bf*k, bf*v, bf*a, bf*b, bf*dy, float*s, float*sa, bf*dr, bf*dw, bf*dk, bf*dv, bf*da, bf*db);
void cuda_backward_tiled256(int B, int T, int H, bf*r, bf*w, bf*k, bf*v, bf*a, bf*b, bf*dy, float*s, float*sa, float*dsb, bf*dr, bf*dw, bf*dk, bf*dv, bf*da, bf*db);

void backward(torch::stable::Tensor &r, torch::stable::Tensor &w, torch::stable::Tensor &k, torch::stable::Tensor &v, torch::stable::Tensor &a, torch::stable::Tensor &b, torch::stable::Tensor &dy,
        torch::stable::Tensor &s, torch::stable::Tensor &sa, torch::stable::Tensor &dr, torch::stable::Tensor &dw, torch::stable::Tensor &dk, torch::stable::Tensor &dv, torch::stable::Tensor &da, torch::stable::Tensor &db, int64_t head_size) {
    STD_TORCH_CHECK(head_size == 64 || head_size == 128 || head_size == 256,
                "head_size must be one of 64, 128, or 256");
    STD_TORCH_CHECK(r.size(3) == head_size, "input head dimension must equal head_size");
    int B = r.sizes()[0], T = r.sizes()[1], H = r.sizes()[2];
    if (head_size == 64) cuda_backward_v3(B, T, H, (bf*)r.mutable_data_ptr(), (bf*)w.mutable_data_ptr(), (bf*)k.mutable_data_ptr(), (bf*)v.mutable_data_ptr(), (bf*)a.mutable_data_ptr(), (bf*)b.mutable_data_ptr(), (bf*)dy.mutable_data_ptr(),
            (float*)s.mutable_data_ptr(), (float*)sa.mutable_data_ptr(), (bf*)dr.mutable_data_ptr(), (bf*)dw.mutable_data_ptr(), (bf*)dk.mutable_data_ptr(), (bf*)dv.mutable_data_ptr(), (bf*)da.mutable_data_ptr(), (bf*)db.mutable_data_ptr());
    else if (head_size == 128) cuda_backward_split(B, T, H, head_size, (bf*)r.mutable_data_ptr(), (bf*)w.mutable_data_ptr(), (bf*)k.mutable_data_ptr(), (bf*)v.mutable_data_ptr(), (bf*)a.mutable_data_ptr(), (bf*)b.mutable_data_ptr(), (bf*)dy.mutable_data_ptr(),
            (float*)s.mutable_data_ptr(), (float*)sa.mutable_data_ptr(), (bf*)dr.mutable_data_ptr(), (bf*)dw.mutable_data_ptr(), (bf*)dk.mutable_data_ptr(), (bf*)dv.mutable_data_ptr(), (bf*)da.mutable_data_ptr(), (bf*)db.mutable_data_ptr());
    else {
        auto dsb = torch::stable::empty_like(sa);
        cuda_backward_tiled256(B, T, H, (bf*)r.mutable_data_ptr(), (bf*)w.mutable_data_ptr(), (bf*)k.mutable_data_ptr(), (bf*)v.mutable_data_ptr(), (bf*)a.mutable_data_ptr(), (bf*)b.mutable_data_ptr(), (bf*)dy.mutable_data_ptr(),
                (float*)s.mutable_data_ptr(), (float*)sa.mutable_data_ptr(), (float*)dsb.mutable_data_ptr(), (bf*)dr.mutable_data_ptr(), (bf*)dw.mutable_data_ptr(), (bf*)dk.mutable_data_ptr(), (bf*)dv.mutable_data_ptr(), (bf*)da.mutable_data_ptr(), (bf*)db.mutable_data_ptr());
    }
}

STABLE_TORCH_LIBRARY_FRAGMENT(flashrwkv2, module) {
  module.def("pretrain_tmix_wkv7_recurrent_forward(Tensor r, Tensor w, Tensor k, Tensor v, Tensor a, Tensor b, Tensor(a!) output, Tensor(b!) boundary, Tensor(c!) state_dot_a, int head_size) -> ()");
  module.def("pretrain_tmix_wkv7_recurrent_backward(Tensor r, Tensor w, Tensor k, Tensor v, Tensor a, Tensor b, Tensor grad_output, Tensor boundary, Tensor state_dot_a, Tensor(a!) grad_r, Tensor(b!) grad_w, Tensor(c!) grad_k, Tensor(d!) grad_v, Tensor(e!) grad_a, Tensor(f!) grad_b, int head_size) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(flashrwkv2, CUDA, module) {
  module.impl("pretrain_tmix_wkv7_recurrent_forward", TORCH_BOX(&forward));
  module.impl("pretrain_tmix_wkv7_recurrent_backward", TORCH_BOX(&backward));
}
