// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to BlinkDL/RWKV-LM
// Canonical source: RWKV-LM RWKV-v7/train_temp/cuda/rwkv7_tmix_tokenshift_bf16_v5.cu
// Source revision: 952102498e9ed367ea0a59ee64106916d474d30f.
// Local adaptation: propagate nonzero initial and returned shift gradients.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <vector>

namespace {

__device__ inline __nv_bfloat162 load_bf16x2(const at::BFloat16* ptr) {
  return *reinterpret_cast<const __nv_bfloat162*>(ptr);
}
__device__ inline void store_bf16x2(at::BFloat16* ptr,
                                    __nv_bfloat162 value) {
  *reinterpret_cast<__nv_bfloat162*>(ptr) = value;
}
__device__ inline void atomic_add_float2(float* ptr, float2 value) {
  atomicAdd(reinterpret_cast<float2*>(ptr), value);
}
inline int64_t ceil_div(int64_t n, int64_t d) { return (n + d - 1) / d; }

__global__ void statetune_tmix_tokenshift_backward_kernel(
    const at::BFloat16* __restrict__ grad_r,
    const at::BFloat16* __restrict__ grad_w,
    const at::BFloat16* __restrict__ grad_k,
    const at::BFloat16* __restrict__ grad_v,
    const at::BFloat16* __restrict__ grad_a,
    const at::BFloat16* __restrict__ grad_g,
    const at::BFloat16* __restrict__ grad_next,
    const at::BFloat16* __restrict__ x,
    const at::BFloat16* __restrict__ initial_shift,
    const at::BFloat16* __restrict__ x_r,
    const at::BFloat16* __restrict__ x_w,
    const at::BFloat16* __restrict__ x_k,
    const at::BFloat16* __restrict__ x_v,
    const at::BFloat16* __restrict__ x_a,
    const at::BFloat16* __restrict__ x_g,
    at::BFloat16* __restrict__ grad_x,
    at::BFloat16* __restrict__ grad_initial, float* __restrict__ grad_x_r,
    float* __restrict__ grad_x_w, float* __restrict__ grad_x_k,
    float* __restrict__ grad_x_v, float* __restrict__ grad_x_a,
    float* __restrict__ grad_x_g, int64_t b_size, int64_t t_size,
    int64_t c_size) {
  const int64_t bc_pair =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t pairs_per_row = c_size / 2;
  if (bc_pair >= b_size * pairs_per_row) return;
  const int64_t b = bc_pair / pairs_per_row;
  const int64_t c = (bc_pair % pairs_per_row) * 2;
  const int64_t base = b * t_size * c_size + c;

  const __nv_bfloat162 one = __floats2bfloat162_rn(1.0f, 1.0f);
  const __nv_bfloat162 pr = load_bf16x2(x_r + c);
  const __nv_bfloat162 pw = load_bf16x2(x_w + c);
  const __nv_bfloat162 pk = load_bf16x2(x_k + c);
  const __nv_bfloat162 pv = load_bf16x2(x_v + c);
  const __nv_bfloat162 pa = load_bf16x2(x_a + c);
  const __nv_bfloat162 pg = load_bf16x2(x_g + c);
  const __nv_bfloat162 mr = __hsub2(one, pr);
  const __nv_bfloat162 mw = __hsub2(one, pw);
  const __nv_bfloat162 mk = __hsub2(one, pk);
  const __nv_bfloat162 mv = __hsub2(one, pv);
  const __nv_bfloat162 ma = __hsub2(one, pa);
  const __nv_bfloat162 mg = __hsub2(one, pg);

  float2 sr = make_float2(0.0f, 0.0f), sw = sr, sk = sr, sv = sr, sa = sr,
         sg = sr;
  for (int64_t t = 0; t < t_size; ++t) {
    const int64_t idx = base + t * c_size;
    const __nv_bfloat162 gr = load_bf16x2(grad_r + idx);
    const __nv_bfloat162 gw = load_bf16x2(grad_w + idx);
    const __nv_bfloat162 gk = load_bf16x2(grad_k + idx);
    const __nv_bfloat162 gv = load_bf16x2(grad_v + idx);
    const __nv_bfloat162 ga = load_bf16x2(grad_a + idx);
    const __nv_bfloat162 gg = load_bf16x2(grad_g + idx);
    // Keep the canonical BF16 products, but accumulate the six independent
    // autograd branches in FP32 before the single BF16 store.  Chaining the
    // additions through __hadd2 rounds after every branch and can exceed the
    // BF16 oracle tolerance for longer sequences even though every product is
    // individually correct.
    float2 dx = make_float2(0.0f, 0.0f);
#define ACCUM_DX(GRAD, COEFFICIENT)                                           \
  do {                                                                        \
    const float2 product =                                                    \
        __bfloat1622float2(__hmul2((GRAD), (COEFFICIENT)));                   \
    dx.x += product.x;                                                        \
    dx.y += product.y;                                                        \
  } while (0)
    ACCUM_DX(gr, mr); ACCUM_DX(gw, mw); ACCUM_DX(gk, mk);
    ACCUM_DX(gv, mv); ACCUM_DX(ga, ma); ACCUM_DX(gg, mg);
    if (t + 1 < t_size) {
      const int64_t next = idx + c_size;
      // Form the recurrent contribution with exactly the same BF16 grouping
      // used by grad_initial below.  A whole sequence and two autograd-linked
      // chunks then see the same rounded boundary gradient before it is added
      // to the six current-token branches.
      __nv_bfloat162 recurrent =
          __hmul2(load_bf16x2(grad_r + next), pr);
      recurrent = __hadd2(
          recurrent, __hmul2(load_bf16x2(grad_w + next), pw));
      recurrent = __hadd2(
          recurrent, __hmul2(load_bf16x2(grad_k + next), pk));
      recurrent = __hadd2(
          recurrent, __hmul2(load_bf16x2(grad_v + next), pv));
      recurrent = __hadd2(
          recurrent, __hmul2(load_bf16x2(grad_a + next), pa));
      recurrent = __hadd2(
          recurrent, __hmul2(load_bf16x2(grad_g + next), pg));
      const float2 recurrent_float = __bfloat1622float2(recurrent);
      dx.x += recurrent_float.x;
      dx.y += recurrent_float.y;
    } else {
      const float2 next = __bfloat1622float2(
          load_bf16x2(grad_next + b * c_size + c));
      dx.x += next.x;
      dx.y += next.y;
    }
    if (t_size == 1) {
      // For the single-token case, match PyTorch's BF16 autograd branch
      // accumulation exactly.  There is no recurrent next-token contribution,
      // and preserving this order avoids a one-ULP cancellation difference
      // after the returned-shift gradient is added.
      __nv_bfloat162 rounded = __hmul2(gr, mr);
      rounded = __hadd2(rounded, __hmul2(gw, mw));
      rounded = __hadd2(rounded, __hmul2(gk, mk));
      rounded = __hadd2(rounded, __hmul2(gv, mv));
      rounded = __hadd2(rounded, __hmul2(ga, ma));
      rounded = __hadd2(rounded, __hmul2(gg, mg));
      rounded = __hadd2(
          rounded, load_bf16x2(grad_next + b * c_size + c));
      store_bf16x2(grad_x + idx, rounded);
    } else {
      store_bf16x2(grad_x + idx, __floats2bfloat162_rn(dx.x, dx.y));
    }
#undef ACCUM_DX

    const __nv_bfloat162 previous =
        t == 0 ? load_bf16x2(initial_shift + b * c_size + c)
               : load_bf16x2(x + idx - c_size);
    const float2 delta =
        __bfloat1622float2(__hsub2(previous, load_bf16x2(x + idx)));
#define ACCUM(NAME, GRAD)                                                     \
  do {                                                                        \
    const float2 f = __bfloat1622float2(GRAD);                                \
    NAME.x += f.x * delta.x;                                                   \
    NAME.y += f.y * delta.y;                                                   \
  } while (0)
    ACCUM(sr, gr); ACCUM(sw, gw); ACCUM(sk, gk); ACCUM(sv, gv);
    ACCUM(sa, ga); ACCUM(sg, gg);
#undef ACCUM
  }

  const int64_t first = base;
  __nv_bfloat162 dinit = __hmul2(load_bf16x2(grad_r + first), pr);
  dinit = __hadd2(dinit, __hmul2(load_bf16x2(grad_w + first), pw));
  dinit = __hadd2(dinit, __hmul2(load_bf16x2(grad_k + first), pk));
  dinit = __hadd2(dinit, __hmul2(load_bf16x2(grad_v + first), pv));
  dinit = __hadd2(dinit, __hmul2(load_bf16x2(grad_a + first), pa));
  dinit = __hadd2(dinit, __hmul2(load_bf16x2(grad_g + first), pg));
  store_bf16x2(grad_initial + b * c_size + c, dinit);
  atomic_add_float2(grad_x_r + c, sr); atomic_add_float2(grad_x_w + c, sw);
  atomic_add_float2(grad_x_k + c, sk); atomic_add_float2(grad_x_v + c, sv);
  atomic_add_float2(grad_x_a + c, sa); atomic_add_float2(grad_x_g + c, sg);
}

__global__ void cast_float_to_bf16_vec2_kernel(
    const float* __restrict__ source, at::BFloat16* __restrict__ target,
    int64_t pairs) {
  const int64_t pair =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pair >= pairs) return;
  const int64_t c = pair * 2;
  store_bf16x2(target + c,
               __floats2bfloat162_rn(source[c], source[c + 1]));
}

}  // namespace

std::vector<torch::Tensor> statetune_tmix_tokenshift_backward_cuda(
    torch::Tensor grad_r, torch::Tensor grad_w, torch::Tensor grad_k,
    torch::Tensor grad_v, torch::Tensor grad_a, torch::Tensor grad_g,
    torch::Tensor grad_next_shift, torch::Tensor x,
    torch::Tensor initial_shift, torch::Tensor x_r, torch::Tensor x_w,
    torch::Tensor x_k, torch::Tensor x_v, torch::Tensor x_a,
    torch::Tensor x_g) {
  auto grad_x = torch::empty_like(x);
  auto grad_initial = torch::empty_like(initial_shift);
  auto fp32 = x.options().dtype(torch::kFloat32);
  std::vector<torch::Tensor> accumulators;
  std::vector<torch::Tensor> coefficient_grads;
  for (int i = 0; i < 6; ++i) {
    accumulators.push_back(torch::zeros({x.size(2)}, fp32));
    coefficient_grads.push_back(torch::empty({x.size(2)}, x.options()));
  }
  constexpr int threads = 256;
  const int64_t bc_pairs = x.size(0) * (x.size(2) / 2);
  const int blocks = static_cast<int>(ceil_div(bc_pairs, threads));
  auto stream = at::cuda::getCurrentCUDAStream();
  statetune_tmix_tokenshift_backward_kernel<<<blocks, threads, 0, stream>>>(
      grad_r.data_ptr<at::BFloat16>(), grad_w.data_ptr<at::BFloat16>(),
      grad_k.data_ptr<at::BFloat16>(), grad_v.data_ptr<at::BFloat16>(),
      grad_a.data_ptr<at::BFloat16>(), grad_g.data_ptr<at::BFloat16>(),
      grad_next_shift.data_ptr<at::BFloat16>(), x.data_ptr<at::BFloat16>(),
      initial_shift.data_ptr<at::BFloat16>(), x_r.data_ptr<at::BFloat16>(),
      x_w.data_ptr<at::BFloat16>(), x_k.data_ptr<at::BFloat16>(),
      x_v.data_ptr<at::BFloat16>(), x_a.data_ptr<at::BFloat16>(),
      x_g.data_ptr<at::BFloat16>(), grad_x.data_ptr<at::BFloat16>(),
      grad_initial.data_ptr<at::BFloat16>(), accumulators[0].data_ptr<float>(),
      accumulators[1].data_ptr<float>(), accumulators[2].data_ptr<float>(),
      accumulators[3].data_ptr<float>(), accumulators[4].data_ptr<float>(),
      accumulators[5].data_ptr<float>(), x.size(0), x.size(1), x.size(2));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const int cast_blocks =
      static_cast<int>(ceil_div(x.size(2) / 2, threads));
  for (int i = 0; i < 6; ++i) {
    cast_float_to_bf16_vec2_kernel<<<cast_blocks, threads, 0, stream>>>(
        accumulators[i].data_ptr<float>(),
        coefficient_grads[i].data_ptr<at::BFloat16>(), x.size(2) / 2);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_x, grad_initial, coefficient_grads[0], coefficient_grads[1],
          coefficient_grads[2], coefficient_grads[3], coefficient_grads[4],
          coefficient_grads[5]};
}
