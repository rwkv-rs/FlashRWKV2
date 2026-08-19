// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to BlinkDL/RWKV-LM
// Canonical source: RWKV-LM RWKV-v7/train_temp/cuda/rwkv7_clampw_v3.cpp
// Source revision: 952102498e9ed367ea0a59ee64106916d474d30f.
// Local adaptation: runtime D=64/128/256 dispatch. D64 keeps clampw-v3,
// D128 follows rwkv7_clampw128_v2, and D256 uses a local warp-tiled extension.

#include <torch/extension.h>

#ifdef _FP32_
    using bf = float;
#else
    #include <cuda_bf16.h>
    using bf = __nv_bfloat16;
#endif

void cuda_forward_v3(int B, int T, int H, bf*r, bf*w, bf*k, bf*v, bf*a, bf*b, bf*y, float*s, float*sa);
void cuda_forward_split(int B, int T, int H, int N, bf*r, bf*w, bf*k, bf*v, bf*a, bf*b, bf*y, float*s, float*sa);

void forward(torch::Tensor &r, torch::Tensor &w, torch::Tensor &k, torch::Tensor &v, torch::Tensor &a, torch::Tensor &b, torch::Tensor &y, torch::Tensor &s, torch::Tensor &sa, int64_t head_size) {
    TORCH_CHECK(head_size == 64 || head_size == 128 || head_size == 256,
                "head_size must be one of 64, 128, or 256");
    TORCH_CHECK(r.size(3) == head_size, "input head dimension must equal head_size");
    int B = r.sizes()[0], T = r.sizes()[1], H = r.sizes()[2];
    if (head_size == 64) cuda_forward_v3(B, T, H, (bf*)r.data_ptr(), (bf*)w.data_ptr(), (bf*)k.data_ptr(), (bf*)v.data_ptr(), (bf*)a.data_ptr(), (bf*)b.data_ptr(), (bf*)y.data_ptr(), (float*)s.data_ptr(), (float*)sa.data_ptr());
    else cuda_forward_split(B, T, H, head_size, (bf*)r.data_ptr(), (bf*)w.data_ptr(), (bf*)k.data_ptr(), (bf*)v.data_ptr(), (bf*)a.data_ptr(), (bf*)b.data_ptr(), (bf*)y.data_ptr(), (float*)s.data_ptr(), (float*)sa.data_ptr());
}

void cuda_backward_v3(int B, int T, int H, bf*r, bf*w, bf*k, bf*v, bf*a, bf*b, bf*dy, float*s, float*sa, bf*dr, bf*dw, bf*dk, bf*dv, bf*da, bf*db);
void cuda_backward_split(int B, int T, int H, int N, bf*r, bf*w, bf*k, bf*v, bf*a, bf*b, bf*dy, float*s, float*sa, bf*dr, bf*dw, bf*dk, bf*dv, bf*da, bf*db);
void cuda_backward_tiled256(int B, int T, int H, bf*r, bf*w, bf*k, bf*v, bf*a, bf*b, bf*dy, float*s, float*sa, float*dsb, bf*dr, bf*dw, bf*dk, bf*dv, bf*da, bf*db);

void backward(torch::Tensor &r, torch::Tensor &w, torch::Tensor &k, torch::Tensor &v, torch::Tensor &a, torch::Tensor &b, torch::Tensor &dy,
        torch::Tensor &s, torch::Tensor &sa, torch::Tensor &dr, torch::Tensor &dw, torch::Tensor &dk, torch::Tensor &dv, torch::Tensor &da, torch::Tensor &db, int64_t head_size) {
    TORCH_CHECK(head_size == 64 || head_size == 128 || head_size == 256,
                "head_size must be one of 64, 128, or 256");
    TORCH_CHECK(r.size(3) == head_size, "input head dimension must equal head_size");
    int B = r.sizes()[0], T = r.sizes()[1], H = r.sizes()[2];
    if (head_size == 64) cuda_backward_v3(B, T, H, (bf*)r.data_ptr(), (bf*)w.data_ptr(), (bf*)k.data_ptr(), (bf*)v.data_ptr(), (bf*)a.data_ptr(), (bf*)b.data_ptr(), (bf*)dy.data_ptr(),
            (float*)s.data_ptr(), (float*)sa.data_ptr(), (bf*)dr.data_ptr(), (bf*)dw.data_ptr(), (bf*)dk.data_ptr(), (bf*)dv.data_ptr(), (bf*)da.data_ptr(), (bf*)db.data_ptr());
    else if (head_size == 128) cuda_backward_split(B, T, H, head_size, (bf*)r.data_ptr(), (bf*)w.data_ptr(), (bf*)k.data_ptr(), (bf*)v.data_ptr(), (bf*)a.data_ptr(), (bf*)b.data_ptr(), (bf*)dy.data_ptr(),
            (float*)s.data_ptr(), (float*)sa.data_ptr(), (bf*)dr.data_ptr(), (bf*)dw.data_ptr(), (bf*)dk.data_ptr(), (bf*)dv.data_ptr(), (bf*)da.data_ptr(), (bf*)db.data_ptr());
    else {
        auto dsb = torch::empty_like(sa);
        cuda_backward_tiled256(B, T, H, (bf*)r.data_ptr(), (bf*)w.data_ptr(), (bf*)k.data_ptr(), (bf*)v.data_ptr(), (bf*)a.data_ptr(), (bf*)b.data_ptr(), (bf*)dy.data_ptr(),
                (float*)s.data_ptr(), (float*)sa.data_ptr(), (float*)dsb.data_ptr(), (bf*)dr.data_ptr(), (bf*)dw.data_ptr(), (bf*)dk.data_ptr(), (bf*)dv.data_ptr(), (bf*)da.data_ptr(), (bf*)db.data_ptr());
    }
}

void register_pretrain_tmix_wkv7_recurrent_bindings(py::module_& module) {
    module.def("pretrain_tmix_wkv7_recurrent_forward", &forward,
               py::arg("r"), py::arg("w"), py::arg("k"), py::arg("v"),
               py::arg("a"), py::arg("b"), py::arg("output"),
               py::arg("boundary"), py::arg("state_dot_a"),
               py::arg("head_size"));
    module.def("pretrain_tmix_wkv7_recurrent_backward", &backward,
               py::arg("r"), py::arg("w"), py::arg("k"), py::arg("v"),
               py::arg("a"), py::arg("b"), py::arg("grad_output"),
               py::arg("boundary"), py::arg("state_dot_a"),
               py::arg("grad_r"), py::arg("grad_w"), py::arg("grad_k"),
               py::arg("grad_v"), py::arg("grad_a"), py::arg("grad_b"),
               py::arg("head_size"));
}
