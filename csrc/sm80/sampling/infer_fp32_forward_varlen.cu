// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright contributors to Rapid-Sampling
// Adapted from Rapid-Sampling revision e0297f7830c3fa581d49ddddddba32f35ea7f733
// and rwkv-rs/vllm-rwkv revision fd440426689f10e240b5761e1a7c82e4c37deb8d.

#include <iostream>
#include <math.h>
#include <assert.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <stdexcept>
#include <utility>
#include <vector>
typedef curandStatePhilox4_32_10_t RAND;

int64_t sampling_state_size_cuda() {
  return static_cast<int64_t>(sizeof(RAND));
}

template <typename T, typename ReduceOp>
__device__ __forceinline__ void warpReduceAll(T& val, ReduceOp op) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    val = op(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
  }
}

template <typename T, typename ReduceOp, int BLOCK_SIZE = 1024,
          bool monotone_sum = false>
__device__ __forceinline__ void blockReduceAll(T& val, ReduceOp op, T identity,
                                               void* buf) {
  T* warpResults = reinterpret_cast<T*>(buf);
  const int lane = threadIdx.x % 32;
  const int warpId = threadIdx.x / 32;
  const int numWarps = (BLOCK_SIZE + 31) / 32;
  warpReduceAll(val, op);
  if (lane == 31) warpResults[warpId] = val;
  __syncthreads();
  T warpVal;
  if constexpr (!monotone_sum) {
    warpVal = (threadIdx.x < numWarps) ? warpResults[threadIdx.x] : identity;
    if (threadIdx.x < 32) warpReduceAll(warpVal, op);
    if (threadIdx.x == 0) warpResults[0] = warpVal;
  } else {
    if (threadIdx.x == 0) {
      warpVal = warpResults[0];
#pragma unroll
      for (int i = 1; i < numWarps; i++) {
        warpVal += warpResults[i];
      }
      warpResults[0] = warpVal;
    }
  }
  __syncthreads();
  val = warpResults[0];
}

template <typename T>
__device__ __forceinline__ T warpInclusiveScan(T val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    T n = __shfl_up_sync(0xFFFFFFFF, val, offset);
    if (threadIdx.x % 32 >= offset) {
      val += n;
    }
  }
  return val;
}

// Block-level inclusive scan - each thread gets sum of itself and all preceding
// threads
template <typename T, int BLOCK_SIZE = 1024>
__device__ __forceinline__ T blockInclusiveScan(T val, void* buf /* shared */,
                                                void* total = nullptr) {
  T* warpSums = reinterpret_cast<T*>(buf);

  const int lane = threadIdx.x % 32;
  const int warpId = threadIdx.x / 32;
  constexpr int numWarps = (BLOCK_SIZE + 31) / 32;

  // Step 1: Inclusive scan within each warp (ok)
  T val1 = warpInclusiveScan(val);

  // Step 2: Last lane of each warp stores its total
  if (lane == 31) {
    warpSums[warpId] = val1;
  }
  __syncthreads();

  // Step 3: First warp does inclusive scan of warp totals
  // if (threadIdx.x < numWarps) {
  //     T warpTotal = warpSums[threadIdx.x];
  //     warpTotal = warpInclusiveScan(warpTotal);
  //     warpSums[threadIdx.x] = warpTotal;
  // }
  // MUST sum this way to ensure numerical MONOTONICITY (not STABILITY)
  if (threadIdx.x == 0) {
    T s = warpSums[0];
#pragma unroll
    for (int i = 1; i < numWarps; i++) {
      s += warpSums[i];
      warpSums[i] = s;
    }
  }
  __syncthreads();

  // Step 4: Add previous warp's prefix to current value
  if (warpId > 0) {
    val1 += warpSums[warpId - 1];
  }
  if (threadIdx.x == BLOCK_SIZE - 1 && total != nullptr) {
    *reinterpret_cast<T*>(total) = val1;
  }
  __syncthreads();
  return val1;
}

// Reduction operation functors
template <typename T>
struct SumOp {
  __device__ __forceinline__ T operator()(T a, T b) const { return a + b; }
  static constexpr T identity() { return T(0); }
};

template <typename T>
struct MaxOp {
  __device__ __forceinline__ T operator()(T a, T b) const { return max(a, b); }
  static constexpr T identity() { return -INFINITY; }  // For float
};

template <typename T>
struct MinOp {
  __device__ __forceinline__ T operator()(T a, T b) const { return min(a, b); }
  static constexpr T identity() { return INFINITY; }  // For float
};

__device__ __forceinline__ float sf(float x) {
  float y = isnan(x) ? 0.0f : x;
  return (isinf(y) ? copysignf(FLT_MAX, y) : y);
}

__global__ void setup_rand_kernel(RAND* states, unsigned long long seed) {
  curand_init(seed, blockIdx.x, 0, &states[blockIdx.x]);
}

at::Tensor setup_sampling_states_cuda(int64_t seed, int64_t num_slots) {
  at::Tensor state =
      at::zeros({num_slots, static_cast<int64_t>(sizeof(RAND))},
                at::TensorOptions().dtype(at::kChar).device(at::kCUDA));
  auto stream = at::cuda::getCurrentCUDAStream();
  setup_rand_kernel<<<static_cast<int>(num_slots), 1, 0, stream>>>(
      reinterpret_cast<RAND*>(state.data_ptr()),
      static_cast<unsigned long long>(seed));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return state;
}

at::Tensor setup_rand(int64_t seed, int64_t B) {
  return setup_sampling_states_cuda(seed, B);
}

static void check_cuda_contiguous_1d(const at::Tensor& tensor, const char* name,
                                     int64_t batch_size, at::ScalarType dtype) {
  if (!tensor.is_cuda() || !tensor.is_contiguous() || tensor.dim() != 1 ||
      tensor.size(0) != batch_size || tensor.scalar_type() != dtype) {
    throw std::invalid_argument(std::string(name) +
                                " must be a contiguous CUDA tensor with shape "
                                "(B,) and the expected dtype");
  }
}

__global__ void validate_sampling_metadata_kernel(
    const int* __restrict__ slot_indices,
    const int* __restrict__ num_active_samples,
    int sample_capacity,
    int num_slots,
    int* __restrict__ status,
    int* __restrict__ seen_slots) {
  __shared__ int active_samples;
  __shared__ int error;
  if (threadIdx.x == 0) {
    active_samples = num_active_samples == nullptr
        ? sample_capacity
        : num_active_samples[0];
    error = active_samples < 0 || active_samples > sample_capacity ? 1 : 0;
    if (error != 0) {
      active_samples = 0;
    }
  }
  __syncthreads();
  for (int slot = static_cast<int>(threadIdx.x); slot < num_slots;
       slot += static_cast<int>(blockDim.x)) {
    seen_slots[slot] = -1;
  }
  __syncthreads();
  for (int row = static_cast<int>(threadIdx.x);
       row < active_samples;
       row += static_cast<int>(blockDim.x)) {
    const int slot = slot_indices[row];
    if (slot < 0 || slot >= num_slots ||
        atomicCAS(seen_slots + slot, -1, row) != -1) {
      atomicExch(&error, 1);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    status[0] = error;
    status[1] = error == 0 ? active_samples : 0;
  }
}

#define BLOCKDIM_X_SAMPLE 1024
__global__ void __launch_bounds__(BLOCKDIM_X_SAMPLE, 1)
    batch_sampling_repetition_temperature_topk_topp_kernel(
        const int B,
        const int T,  // should be 1 typically; may not be 1 if full output is
                      // obtained
        const int V,  // vocabulary size, 60,000 ~ 120,000
        const float* __restrict__ logits,  // (B, V) if T == 1; If T != 1, only
                                           // logits[:, T-1, :] is read. This
                                           // avoids another copying operation
        float* __restrict__ penalties,     // (B, V), can set some to -INF for
                                           // masking
        const int* __restrict__ penalty_indices,  // optional (B,), maps batch
                                                  // row to penalties row
        int* __restrict__ outputs,                // (B,)
        float* __restrict__ output_logprobs,      // (B,)
        RAND* __restrict__ states,                // random state, typedef
                                    // curandStatePhilox4_32_10_t RAND;
        float* __restrict__ probs,  // probs (in L2 cache)
        const float* __restrict__ presence_penalties,
        const float* __restrict__ repetition_penalties,
        const float* __restrict__ penalty_decays,
        const float* __restrict__ temperatures, const int* __restrict__ top_ks,
        const float* __restrict__ top_ps, const float scalar_presence_penalty,
        const float scalar_repetition_penalty, const float scalar_penalty_decay,
        const float scalar_temperature, const int scalar_top_k,
        const float scalar_top_p,
        const int* __restrict__ sampling_status) {
  const int b = blockIdx.x;
  const int d = blockDim.x;
  const int t = threadIdx.x;
  const int w = t / 32;
  const int l = t % 32;
  if (sampling_status != nullptr &&
      (sampling_status[0] != 0 || b >= sampling_status[1])) {
    if (t == 0) {
      outputs[b] = -1;
      if (output_logprobs != nullptr) {
        output_logprobs[b] = -INFINITY;
      }
    }
    return;
  }
  // constexpr int W = (BLOCKDIM_X_SAMPLE + 31) / 32;
  __shared__ __align__(256) char reduce_buf[256];
  __builtin_assume(BLOCKDIM_X_SAMPLE == d);
  __builtin_assume(V % 4 == 0);
  __builtin_assume(V <= 1048576);
  const int V4 = V / 4;
  float4 l4, p4;
  const float presence_penalty = presence_penalties == nullptr
                                     ? scalar_presence_penalty
                                     : presence_penalties[b];
  const float repetition_penalty = repetition_penalties == nullptr
                                       ? scalar_repetition_penalty
                                       : repetition_penalties[b];
  const float penalty_decay =
      penalty_decays == nullptr ? scalar_penalty_decay : penalty_decays[b];
  const float temperature = fminf(
      fmaxf(temperatures == nullptr ? scalar_temperature : temperatures[b],
            0.001f),
      1000.0f);
  int top_k = top_ks == nullptr ? scalar_top_k : top_ks[b];
  float top_p =
      fminf(fmaxf(top_ps == nullptr ? scalar_top_p : top_ps[b], 0.0f), 1.0f);
  if (top_k <= 0 || top_k > V) top_k = V;
  if (top_p == 0.0f) {
    top_k = 1;
    top_p = 1.0f;
  }
  const float log2_inv_temp = float(M_LOG2E) / temperature;

  logits += (b * T + (T - 1)) * V;  // B T V
  const int penalty_row = penalty_indices == nullptr ? b : penalty_indices[b];
  if (penalties != nullptr) penalties += penalty_row * V;  // slot rows V
  outputs += b;                                          // B
  if (output_logprobs != nullptr) output_logprobs += b;  // B
  states += penalty_row;                                 // request slot
  probs += (b * T + (T - 1)) * V;                        // B T V

  float maxu = -INFINITY;
  for (int i = t; i < V4; i += d) {
    l4 = ((float4*)logits)[i];
    p4 = penalties == nullptr ? make_float4(0.f, 0.f, 0.f, 0.f)
                              : ((float4*)penalties)[i];
#pragma unroll
    for (int j = 0; j < 4; j++) {
      float& fl = ((float*)&l4)[j];
      // if (i*4+j < 3){
      //     P0i(i*4+j);
      //     P0f(fl);
      // }
      float& fp = ((float*)&p4)[j];
      fl = sf((sf(fl) - fp) * log2_inv_temp);
      maxu = max(maxu, fl);
      // ((float*)&l4)[j] = fr;
    }
    ((float4*)probs)[i] = l4;
  }
  blockReduceAll(maxu, MaxOp<float>{}, MaxOp<float>::identity(), reduce_buf);
  __syncthreads();
  float exp_denom = 0;
  for (int i = t; i < V4; i += d) {
    l4 = ((float4*)probs)[i];
    float em = 0.f;
#pragma unroll
    for (int j = 0; j < 4; j++) {
      float& fr = ((float*)&l4)[j];
      em += exp2f(fr - maxu);
    }
    exp_denom += em;
  }
  blockReduceAll(exp_denom, SumOp<float>{}, SumOp<float>::identity(),
                 reduce_buf);
  __syncthreads();
  float pmax = -INFINITY;
  float pmin = +INFINITY;
  for (int i = t; i < V4; i += d) {
    l4 = ((float4*)probs)[i];
#pragma unroll
    for (int j = 0; j < 4; j++) {
      float& fr = ((float*)&l4)[j];
      fr = exp2f(fr - maxu) / exp_denom;
      pmax = max(pmax, fr);
      pmin = min(pmin, fr);
      // ((float*)&l4)[j] = fr;
    }
    ((float4*)probs)[i] = l4;
  }
  blockReduceAll(pmax, MaxOp<float>{}, MaxOp<float>::identity(), reduce_buf);
  __syncthreads();
  blockReduceAll(pmin, MinOp<float>{}, MinOp<float>::identity(), reduce_buf);
  __syncthreads();

  // if(t==0) P0f(pmax);
  unsigned left = __float_as_uint(pmin), right = __float_as_uint(pmax) + 1;

  uint4 cnt = {.x = (unsigned)V, .y = 0, .z = 0, .w = 0};
  l4 = {.x = 1, .y = 0, .z = 0, .w = 0};
  uint4 pivot;
  while ((cnt.x > top_k || l4.x > top_p) && left < right - 1) {
    // if(t==0){
    //     P0i(top_k);
    //     P0i(left);
    //     P0i(right);
    //     P0i(cnt.x);
    //     printf("\n");
    // }
    pivot.x = left;
    pivot.z = (left + right) / 2;
    pivot.y = (left + pivot.z) / 2;
    pivot.w = (pivot.z + right) / 2;
    l4.y = l4.z = l4.w = 0;
    cnt.y = cnt.z = cnt.w = 0;
    for (int i = t; i < V4; i += d) {
      p4 = ((float4*)probs)[i];
#pragma unroll
      for (int j = 0; j < 4; j++) {
        float& p = ((float*)&p4)[j];
        if (p >= __uint_as_float(pivot.y)) {
          cnt.y++;
          l4.y += p;
        }
        if (p >= __uint_as_float(pivot.z)) {
          cnt.z++;
          l4.z += p;
        }
        if (p >= __uint_as_float(pivot.w)) {
          cnt.w++;
          l4.w += p;
        }
      }
    }
    blockReduceAll(cnt.y, SumOp<unsigned>{}, SumOp<unsigned>::identity(),
                   reduce_buf);
    __syncthreads();
    blockReduceAll<float, SumOp<float>, BLOCKDIM_X_SAMPLE, true>(
        l4.y, SumOp<float>{}, SumOp<float>::identity(), reduce_buf);
    __syncthreads();
    if (cnt.y < top_k && l4.y < top_p) {
      left = pivot.x;
      right = pivot.y;
      // cnt.x = cnt.x;
      // l4.x = l4.x;
      continue;
    }
    blockReduceAll(cnt.z, SumOp<unsigned>{}, SumOp<unsigned>::identity(),
                   reduce_buf);
    __syncthreads();
    blockReduceAll<float, SumOp<float>, BLOCKDIM_X_SAMPLE, true>(
        l4.z, SumOp<float>{}, SumOp<float>::identity(), reduce_buf);
    __syncthreads();
    if (cnt.z < top_k && l4.z < top_p) {
      left = pivot.y;
      right = pivot.z;
      cnt.x = cnt.y;
      l4.x = l4.y;
      continue;
    }
    blockReduceAll(cnt.w, SumOp<unsigned>{}, SumOp<unsigned>::identity(),
                   reduce_buf);
    __syncthreads();
    blockReduceAll<float, SumOp<float>, BLOCKDIM_X_SAMPLE, true>(
        l4.w, SumOp<float>{}, SumOp<float>::identity(), reduce_buf);
    __syncthreads();
    if (cnt.w < top_k && l4.w < top_p) {
      left = pivot.z;
      right = pivot.w;
      cnt.x = cnt.z;
      l4.x = l4.z;
      continue;
    }
    left = pivot.w;
    // right = right;
    cnt.x = cnt.w;
    l4.x = l4.w;
  }
  // return left
  float threshold = __uint_as_float(left);
  // if(t==0) P0f(threshold);
  // 5. recompute (read once)
  float gtp = 0;
  unsigned eqk = 0, gtk = 0;
  __shared__ float /* seqp, */ sgtp;
  __shared__ unsigned seqk, sgtk;

  for (int i = t; i < V4; i += d) {
    p4 = ((float4*)probs)[i];
#pragma unroll
    for (int j = 0; j < 4; j++) {
      float& p = ((float*)&p4)[j];
      if (p == threshold) eqk++;
      if (p > threshold) {
        gtk++;
        gtp += p;
      }
    }
  }
  // s: shared all
  // c: cumulative
  // -: per thread
  // __syncthreads();
  float cgtp = blockInclusiveScan(gtp, reduce_buf, &sgtp);
  __syncthreads();
  unsigned ceqk = blockInclusiveScan(eqk, reduce_buf, &seqk);
  __syncthreads();
  unsigned cgtk = blockInclusiveScan(gtk, reduce_buf, &sgtk);
  __syncthreads();
  // if(t==0) P0f(sgtp);
  // if(t==0) P0i(seqk);
  // if(t==0) P0i(sgtk);

  // compute compensation
  // seqk == total number of tokens that equals threshold
  // _gtp + threshold * _eqk == _eqp
  // (top_p - sgtp) == delta_p
  // delta_p / seqp
  unsigned neqk = seqk;
  float comp = 1.0f;
  if (neqk > 0) {
    comp = min(sf((top_p - sgtp) / (threshold * neqk)), comp);
    comp = min(sf(float(top_k - sgtk) / neqk), comp);
    comp = max(comp, 0.0f);
  }

  // 6. Yield sampled tokens
  __shared__ float randp, sum_p;
  __shared__ float4 rand4;
  __shared__ int idxt;
  float actual_p = gtp + (threshold * eqk) * comp;
  __syncthreads();
  float cumu_p = blockInclusiveScan(actual_p, reduce_buf, &sum_p);
  __syncthreads();
  if (t == 0) {
    idxt = 0;
    rand4 = curand_uniform4(states);
    randp = sum_p * rand4.x;  // only once
  }
  __syncthreads();

  bool u = (randp <= cumu_p);
  // at last thread: randp = sum_p * rand4.x < cumu_p == sum_p, u == 1
  if (l == 31) ((unsigned*)reduce_buf)[w] = u;
  __syncthreads();
  bool u_ = __shfl_up_sync(0xffffffff, u, 1);
  if (t == 0)
    u_ = 0;
  else if (l == 0)
    u_ = ((unsigned*)reduce_buf)[w - 1];
  __syncthreads();

  if (u != u_) idxt = t;
  __syncthreads();

  // a sub-tile (of no more than 1024)
  int idn = idxt * 4 + (t / 4) * 4 * d + (t % 4);
  // .... .... (idxt) |||| .... .... .... |||| .... .... .... |||| ....
  float o0 = (idn < V) ? (probs[idn]) : 0;
  float o = (o0 < threshold) ? 0 : (o0 == threshold) ? (o0 * comp) : o0;

  __shared__ float sum_o;
  float cumu_o = blockInclusiveScan(o, reduce_buf, &sum_o);  // monotone
  __syncthreads();
  float rand_2 = sum_o * rand4.y;
  u = (rand_2 <= cumu_o);
  // at last thread: cumu_o == sum_o, rand4.y < 1, sum_o * rand4.y < cumu_o, u
  // == 1
  if (l == 31) ((unsigned*)reduce_buf)[w] = u;
  // u: current u_: prev
  // at first thread: u_ == 0
  u_ = __shfl_up_sync(0xffffffff, u, 1);
  __syncthreads();
  if (t == 0)
    u_ = 0;
  else if (l == 0)
    u_ = ((unsigned*)reduce_buf)[w - 1];
  __syncthreads();

  // write idn
  __shared__ int out_id;
  __shared__ float out_logprob;
  if (u != u_) {
    out_id = (idn < V) ? idn : 0;
    if (output_logprobs != nullptr) out_logprob = logf(o) - logf(sum_p);
  }
  __syncthreads();
  idn = out_id;
  if (t == 0) {
    *outputs = idn;
    if (output_logprobs != nullptr) *output_logprobs = out_logprob;
  }
  // 7. Update penalties
  if (penalties != nullptr) for (int i = t; i < V4; i += d) {
    p4 = ((float4*)penalties)[i];
#pragma unroll
    for (int j = 0; j < 4; j++) {
      float& p = ((float*)&p4)[j];
      int idp = i * 4 + j;
      const bool token_seen = p != 0.0f || signbit(p);
      const bool sampled_token = idn == idp;
      p = fmaf(p, penalty_decay,
               sampled_token
                   ? repetition_penalty + (token_seen ? 0.0f : presence_penalty)
                   : 0.0f);
      // `-0.0` marks a generated token whose accumulated penalty cancels to
      // zero. It is numerically a no-op when subtracting penalties from logits,
      // but retains the presence-penalty state for later occurrences.
      if ((token_seen || sampled_token) && p == 0.0f) {
        p = -0.0f;
      }
    }
    ((float4*)penalties)[i] = p4;
  }
}

std::vector<at::Tensor> batch_sampling_repetition_temperature_topk_topp(
    at::Tensor& logits, at::Tensor& penalties, at::Tensor& states,
    double presence_penalty, double repetition_penalty, double penalty_decay,
    double temperature, int64_t top_k, double top_p, bool return_logprobs) {
  int B, T, V;
  if (logits.dtype() != at::kFloat) {
    throw std::invalid_argument(
        "Logits tensor must be of type float32 (FP32), got " +
        std::string(logits.dtype().name()) + " !\n");
  }
  V = logits.size(-1);
  B = (penalties.dim() == 2) ? penalties.size(0) : 1;
  T = (logits.dim() == 3) ? logits.size(1) : 1;
  if (!(V > 0 && V <= 1048576 && V % 4 == 0)) {
    throw std::invalid_argument(
        "Vocabulary size must be multiple of 4, and no larger than 1048576, "
        "got " +
        std::to_string(V) + " !\n");
  }
  if (!(B > 0 && T > 0)) {
    throw std::invalid_argument(
        "B and T must be positive, got B=" + std::to_string(B) +
        ", T=" + std::to_string(T) + " !\n");
  }
  if (!(temperature >= 0.001 && temperature <= 1000)) {
    throw std::invalid_argument("Temperature outside range, got " +
                                std::to_string(temperature) +
                                ", expect [0.001, 1000]!\n");
  }
  if (top_k <= 0 || top_k > V) top_k = V;
  if (top_p < 0 || top_p > 1) top_p = 1;
  if (top_p == 0) {
    top_k = 1;
    top_p = 1;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  auto probs = at::empty(
      {B, V}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));
  if (B * V * 4 <= 4194304) {
    cudaStreamAttrValue stream_attribute;
    stream_attribute.accessPolicyWindow.base_ptr = probs.data_ptr();
    stream_attribute.accessPolicyWindow.num_bytes = B * V * 4;
    stream_attribute.accessPolicyWindow.hitRatio = 1;
    stream_attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    stream_attribute.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
    cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow,
                           &stream_attribute);
  }
  auto out =
      at::empty({B}, at::TensorOptions().dtype(at::kInt).device(at::kCUDA));
  std::vector<at::Tensor> outputs = {out};
  float* output_logprobs = nullptr;
  if (return_logprobs) {
    auto out_logprobs =
        at::empty({B}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));
    output_logprobs = (float*)out_logprobs.data_ptr();
    outputs.push_back(out_logprobs);
  }

  batch_sampling_repetition_temperature_topk_topp_kernel<<<B, 1024, 0,
                                                           stream>>>(
      B, T, V, (float*)logits.data_ptr(), (float*)penalties.data_ptr(), nullptr,
      (int*)out.data_ptr(), output_logprobs, (RAND*)states.data_ptr(),
      (float*)probs.data_ptr(), nullptr, nullptr, nullptr, nullptr, nullptr,
      nullptr, (float)presence_penalty, (float)repetition_penalty,
      (float)penalty_decay, (float)temperature, (int)top_k, (float)top_p,
      nullptr);
  return outputs;
}

std::vector<at::Tensor> batch_sampling_repetition_temperature_topk_topp_indexed(
    at::Tensor& logits, at::Tensor& penalties, at::Tensor& penalty_indices,
    at::Tensor& states, double presence_penalty, double repetition_penalty,
    double penalty_decay, double temperature, int64_t top_k, double top_p,
    bool return_logprobs) {
  int B, T, V;
  if (logits.dtype() != at::kFloat) {
    throw std::invalid_argument(
        "Logits tensor must be of type float32 (FP32), got " +
        std::string(logits.dtype().name()) + " !\n");
  }
  if (penalties.dtype() != at::kFloat) {
    throw std::invalid_argument(
        "Penalties tensor must be of type float32 (FP32), got " +
        std::string(penalties.dtype().name()) + " !\n");
  }
  if (penalty_indices.dtype() != at::kInt) {
    throw std::invalid_argument(
        "Penalty indices tensor must be of type int32, got " +
        std::string(penalty_indices.dtype().name()) + " !\n");
  }
  V = logits.size(-1);
  B = (logits.dim() == 3) ? logits.size(0)
                          : (logits.dim() == 2 ? logits.size(0) : 1);
  T = (logits.dim() == 3) ? logits.size(1) : 1;
  if (!(V > 0 && V <= 1048576 && V % 4 == 0)) {
    throw std::invalid_argument(
        "Vocabulary size must be multiple of 4, and no larger than 1048576, "
        "got " +
        std::to_string(V) + " !\n");
  }
  if (!(B > 0 && T > 0)) {
    throw std::invalid_argument(
        "B and T must be positive, got B=" + std::to_string(B) +
        ", T=" + std::to_string(T) + " !\n");
  }
  if (!(penalties.dim() == 2 && penalties.size(1) == V)) {
    throw std::invalid_argument(
        "Penalties tensor must have shape (rows, V), got dim=" +
        std::to_string(penalties.dim()) +
        " and V=" + std::to_string(penalties.size(-1)) + " !\n");
  }
  if (!(penalty_indices.dim() == 1 && penalty_indices.size(0) == B &&
        penalty_indices.is_contiguous())) {
    throw std::invalid_argument(
        "Penalty indices tensor must be contiguous with shape (B,), got dim=" +
        std::to_string(penalty_indices.dim()) +
        " and rows=" + std::to_string(penalty_indices.size(0)) + " !\n");
  }
  if (!(temperature >= 0.001 && temperature <= 1000)) {
    throw std::invalid_argument("Temperature outside range, got " +
                                std::to_string(temperature) +
                                ", expect [0.001, 1000]!\n");
  }
  if (top_k <= 0 || top_k > V) top_k = V;
  if (top_p < 0 || top_p > 1) top_p = 1;
  if (top_p == 0) {
    top_k = 1;
    top_p = 1;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  auto probs = at::empty(
      {B, V}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));
  if (B * V * 4 <= 4194304) {
    cudaStreamAttrValue stream_attribute;
    stream_attribute.accessPolicyWindow.base_ptr = probs.data_ptr();
    stream_attribute.accessPolicyWindow.num_bytes = B * V * 4;
    stream_attribute.accessPolicyWindow.hitRatio = 1;
    stream_attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    stream_attribute.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
    cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow,
                           &stream_attribute);
  }
  auto out =
      at::empty({B}, at::TensorOptions().dtype(at::kInt).device(at::kCUDA));
  std::vector<at::Tensor> outputs = {out};
  float* output_logprobs = nullptr;
  if (return_logprobs) {
    auto out_logprobs =
        at::empty({B}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));
    output_logprobs = (float*)out_logprobs.data_ptr();
    outputs.push_back(out_logprobs);
  }

  batch_sampling_repetition_temperature_topk_topp_kernel<<<B, 1024, 0,
                                                           stream>>>(
      B, T, V, (float*)logits.data_ptr(), (float*)penalties.data_ptr(),
      (int*)penalty_indices.data_ptr(), (int*)out.data_ptr(), output_logprobs,
      (RAND*)states.data_ptr(), (float*)probs.data_ptr(), nullptr, nullptr,
      nullptr, nullptr, nullptr, nullptr, (float)presence_penalty,
      (float)repetition_penalty, (float)penalty_decay, (float)temperature,
      (int)top_k, (float)top_p, nullptr);
  return outputs;
}

std::vector<at::Tensor>
batch_sampling_repetition_temperature_topk_topp_per_request_indexed(
    at::Tensor& logits, at::Tensor& penalties, at::Tensor& penalty_indices,
    at::Tensor& states, at::Tensor& presence_penalties,
    at::Tensor& repetition_penalties, at::Tensor& penalty_decays,
    at::Tensor& temperatures, at::Tensor& top_ks, at::Tensor& top_ps,
    bool return_logprobs) {
  const int V = logits.size(-1);
  const int B = logits.dim() >= 2 ? logits.size(0) : 1;
  const int T = logits.dim() == 3 ? logits.size(1) : 1;
  if (logits.dtype() != at::kFloat || penalties.dtype() != at::kFloat) {
    throw std::invalid_argument("Logits and penalties must be float32");
  }
  if (!(V > 0 && V <= 1048576 && V % 4 == 0) || B <= 0 || T <= 0) {
    throw std::invalid_argument("Invalid rapid-sampling logits shape");
  }
  if (!(penalties.dim() == 2 && penalties.size(1) == V)) {
    throw std::invalid_argument("Penalties must have shape (rows, V)");
  }
  check_cuda_contiguous_1d(penalty_indices, "penalty_indices", B, at::kInt);
  check_cuda_contiguous_1d(presence_penalties, "presence_penalties", B,
                           at::kFloat);
  check_cuda_contiguous_1d(repetition_penalties, "repetition_penalties", B,
                           at::kFloat);
  check_cuda_contiguous_1d(penalty_decays, "penalty_decays", B, at::kFloat);
  check_cuda_contiguous_1d(temperatures, "temperatures", B, at::kFloat);
  check_cuda_contiguous_1d(top_ks, "top_ks", B, at::kInt);
  check_cuda_contiguous_1d(top_ps, "top_ps", B, at::kFloat);

  auto stream = at::cuda::getCurrentCUDAStream();
  auto probs = at::empty(
      {B, V}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));
  auto out =
      at::empty({B}, at::TensorOptions().dtype(at::kInt).device(at::kCUDA));
  std::vector<at::Tensor> outputs = {out};
  float* output_logprobs = nullptr;
  if (return_logprobs) {
    auto out_logprobs =
        at::empty({B}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));
    output_logprobs = (float*)out_logprobs.data_ptr();
    outputs.push_back(out_logprobs);
  }
  batch_sampling_repetition_temperature_topk_topp_kernel<<<B, 1024, 0,
                                                           stream>>>(
      B, T, V, (float*)logits.data_ptr(), (float*)penalties.data_ptr(),
      (int*)penalty_indices.data_ptr(), (int*)out.data_ptr(), output_logprobs,
      (RAND*)states.data_ptr(), (float*)probs.data_ptr(),
      (float*)presence_penalties.data_ptr(),
      (float*)repetition_penalties.data_ptr(),
      (float*)penalty_decays.data_ptr(), (float*)temperatures.data_ptr(),
      (int*)top_ks.data_ptr(), (float*)top_ps.data_ptr(), 0.0f, 0.0f, 1.0f,
      1.0f, V, 1.0f, nullptr);
  return outputs;
}

__global__ void __launch_bounds__(BLOCKDIM_X_SAMPLE, 1)
    batch_sampling_temperature_topk_topp_kernel(
        const int B,
        const int T,  // should be 1 typically; may not be 1 if full output is
                      // obtained
        const int V,  // vocabulary size, 60,000 ~ 120,000
        const float* __restrict__ logits,  // (B, V) if T == 1; If T != 1, only
                                           // logits[:, T-1, :] is read. This
                                           // avoids another copying operation
        int* __restrict__ outputs,         // (B,)
        float* __restrict__ output_logprobs,  // (B,)
        RAND* __restrict__ states,            // random state, typedef
                                    // curandStatePhilox4_32_10_t RAND;
        float* __restrict__ probs,  // probs (in L2 cache)
        const int* __restrict__ slot_indices,
        const float log2e_inv_temp, const int top_k, const float top_p,
        const int* __restrict__ sampling_status) {
  const int b = blockIdx.x;
  const int d = blockDim.x;
  const int t = threadIdx.x;
  const int w = t / 32;
  const int l = t % 32;
  if (sampling_status != nullptr &&
      (sampling_status[0] != 0 || b >= sampling_status[1])) {
    if (t == 0) {
      outputs[b] = -1;
      if (output_logprobs != nullptr) {
        output_logprobs[b] = -INFINITY;
      }
    }
    return;
  }
  __shared__ __align__(256) char reduce_buf[256];
  __builtin_assume(BLOCKDIM_X_SAMPLE == d);
  __builtin_assume(V % 4 == 0);
  __builtin_assume(V <= 1048576);
  __builtin_assume(log2e_inv_temp > 0.f);
  const int V4 = V / 4;
  float4 l4, p4;

  logits += (b * T + (T - 1)) * V;                       // B T V
  outputs += b;                                          // B
  if (output_logprobs != nullptr) output_logprobs += b;  // B
  states += slot_indices == nullptr ? b : slot_indices[b];  // request slot
  probs += (b * T + (T - 1)) * V;                        // B T V

  float maxu = -INFINITY;
  for (int i = t; i < V4; i += d) {
    l4 = ((float4*)logits)[i];
#pragma unroll
    for (int j = 0; j < 4; j++) {
      float& fl = ((float*)&l4)[j];
      fl = sf(fl * log2e_inv_temp);
      maxu = max(maxu, fl);
    }
    ((float4*)probs)[i] = l4;
  }
  blockReduceAll(maxu, MaxOp<float>{}, MaxOp<float>::identity(), reduce_buf);
  __syncthreads();
  float exp_denom = 0;
  for (int i = t; i < V4; i += d) {
    l4 = ((float4*)probs)[i];
    float em = 0.f;
#pragma unroll
    for (int j = 0; j < 4; j++) {
      float& fr = ((float*)&l4)[j];
      em += exp2f(fr - maxu);
    }
    exp_denom += em;
  }
  blockReduceAll(exp_denom, SumOp<float>{}, SumOp<float>::identity(),
                 reduce_buf);
  __syncthreads();
  float pmax = -INFINITY;
  float pmin = +INFINITY;
  for (int i = t; i < V4; i += d) {
    l4 = ((float4*)probs)[i];
#pragma unroll
    for (int j = 0; j < 4; j++) {
      float& fr = ((float*)&l4)[j];
      fr = exp2f(fr - maxu) / exp_denom;
      pmax = max(pmax, fr);
      pmin = min(pmin, fr);
      // ((float*)&l4)[j] = fr;
    }
    ((float4*)probs)[i] = l4;
  }
  blockReduceAll(pmax, MaxOp<float>{}, MaxOp<float>::identity(), reduce_buf);
  __syncthreads();
  blockReduceAll(pmin, MinOp<float>{}, MinOp<float>::identity(), reduce_buf);
  __syncthreads();

  // if(t==0) P0f(pmax);
  unsigned left = __float_as_uint(pmin), right = __float_as_uint(pmax) + 1;

  uint4 cnt = {.x = (unsigned)V, .y = 0, .z = 0, .w = 0};
  l4 = {.x = 1, .y = 0, .z = 0, .w = 0};
  uint4 pivot;
  while ((cnt.x > top_k || l4.x > top_p) && left < right - 1) {
    pivot.x = left;
    pivot.z = (left + right) / 2;
    pivot.y = (left + pivot.z) / 2;
    pivot.w = (pivot.z + right) / 2;
    l4.y = l4.z = l4.w = 0;
    cnt.y = cnt.z = cnt.w = 0;
    for (int i = t; i < V4; i += d) {
      p4 = ((float4*)probs)[i];
#pragma unroll
      for (int j = 0; j < 4; j++) {
        float& p = ((float*)&p4)[j];
        if (p >= __uint_as_float(pivot.y)) {
          cnt.y++;
          l4.y += p;
        }
        if (p >= __uint_as_float(pivot.z)) {
          cnt.z++;
          l4.z += p;
        }
        if (p >= __uint_as_float(pivot.w)) {
          cnt.w++;
          l4.w += p;
        }
      }
    }
    blockReduceAll(cnt.y, SumOp<unsigned>{}, SumOp<unsigned>::identity(),
                   reduce_buf);
    __syncthreads();
    blockReduceAll<float, SumOp<float>, BLOCKDIM_X_SAMPLE, true>(
        l4.y, SumOp<float>{}, SumOp<float>::identity(), reduce_buf);
    __syncthreads();
    if (cnt.y < top_k && l4.y < top_p) {
      left = pivot.x;
      right = pivot.y;
      // cnt.x = cnt.x;
      // l4.x = l4.x;
      continue;
    }
    blockReduceAll(cnt.z, SumOp<unsigned>{}, SumOp<unsigned>::identity(),
                   reduce_buf);
    __syncthreads();
    blockReduceAll<float, SumOp<float>, BLOCKDIM_X_SAMPLE, true>(
        l4.z, SumOp<float>{}, SumOp<float>::identity(), reduce_buf);
    __syncthreads();
    if (cnt.z < top_k && l4.z < top_p) {
      left = pivot.y;
      right = pivot.z;
      cnt.x = cnt.y;
      l4.x = l4.y;
      continue;
    }
    blockReduceAll(cnt.w, SumOp<unsigned>{}, SumOp<unsigned>::identity(),
                   reduce_buf);
    __syncthreads();
    blockReduceAll<float, SumOp<float>, BLOCKDIM_X_SAMPLE, true>(
        l4.w, SumOp<float>{}, SumOp<float>::identity(), reduce_buf);
    __syncthreads();
    if (cnt.w < top_k && l4.w < top_p) {
      left = pivot.z;
      right = pivot.w;
      cnt.x = cnt.z;
      l4.x = l4.z;
      continue;
    }
    left = pivot.w;
    // right = right;
    cnt.x = cnt.w;
    l4.x = l4.w;
  }
  // return left
  float threshold = __uint_as_float(left);
  // if(t==0) P0f(threshold);
  // 5. recompute (read once)
  float gtp = 0;
  unsigned eqk = 0, gtk = 0;
  __shared__ float /* seqp, */ sgtp;
  __shared__ unsigned seqk, sgtk;

  for (int i = t; i < V4; i += d) {
    p4 = ((float4*)probs)[i];
#pragma unroll
    for (int j = 0; j < 4; j++) {
      float& p = ((float*)&p4)[j];
      if (p == threshold) eqk++;
      if (p > threshold) {
        gtk++;
        gtp += p;
      }
    }
  }
  // s: shared all
  // c: cumulative
  // -: per thread
  // __syncthreads();
  float cgtp = blockInclusiveScan(gtp, reduce_buf, &sgtp);
  __syncthreads();
  unsigned ceqk = blockInclusiveScan(eqk, reduce_buf, &seqk);
  __syncthreads();
  unsigned cgtk = blockInclusiveScan(gtk, reduce_buf, &sgtk);
  __syncthreads();
  // if(t==0) P0f(sgtp);
  // if(t==0) P0i(seqk);
  // if(t==0) P0i(sgtk);

  // compute compensation
  // seqk == total number of tokens that equals threshold
  // _gtp + threshold * _eqk == _eqp
  // (top_p - sgtp) == delta_p
  // delta_p / seqp
  unsigned neqk = seqk;
  float comp = 1.0f;
  if (neqk > 0) {
    comp = min(sf((top_p - sgtp) / (threshold * neqk)), comp);
    comp = min(sf(float(top_k - sgtk) / neqk), comp);
    comp = max(comp, 0.0f);
  }

  // 6. Yield sampled tokens
  __shared__ float randp, sum_p;
  __shared__ float4 rand4;
  __shared__ int idxt;
  float actual_p = gtp + (threshold * eqk) * comp;
  __syncthreads();
  float cumu_p = blockInclusiveScan(actual_p, reduce_buf, &sum_p);
  __syncthreads();
  if (t == 0) {
    idxt = 0;
    rand4 = curand_uniform4(states);
    randp = sum_p * rand4.x;  // only once
  }
  __syncthreads();

  bool u = (randp <= cumu_p);
  // at last thread: randp = sum_p * rand4.x < cumu_p == sum_p, u == 1
  if (l == 31) ((unsigned*)reduce_buf)[w] = u;
  __syncthreads();
  bool u_ = __shfl_up_sync(0xffffffff, u, 1);
  if (t == 0)
    u_ = 0;
  else if (l == 0)
    u_ = ((unsigned*)reduce_buf)[w - 1];
  __syncthreads();

  if (u != u_) idxt = t;
  __syncthreads();

  // a sub-tile (of no more than 1024)
  int idn = idxt * 4 + (t / 4) * 4 * d + (t % 4);
  // .... .... (idxt) |||| .... .... .... |||| .... .... .... |||| ....
  float o0 = (idn < V) ? (probs[idn]) : 0;
  float o = (o0 < threshold) ? 0 : (o0 == threshold) ? (o0 * comp) : o0;

  __shared__ float sum_o;
  float cumu_o = blockInclusiveScan(o, reduce_buf, &sum_o);  // monotone
  __syncthreads();
  float rand_2 = sum_o * rand4.y;
  u = (rand_2 <= cumu_o);
  // at last thread: cumu_o == sum_o, rand4.y < 1, sum_o * rand4.y < cumu_o, u
  // == 1
  if (l == 31) ((unsigned*)reduce_buf)[w] = u;
  // u: current u_: prev
  // at first thread: u_ == 0
  u_ = __shfl_up_sync(0xffffffff, u, 1);
  __syncthreads();
  if (t == 0)
    u_ = 0;
  else if (l == 0)
    u_ = ((unsigned*)reduce_buf)[w - 1];
  __syncthreads();

  // write idn
  __shared__ int out_id;
  __shared__ float out_logprob;
  if (u != u_) {
    out_id = (idn < V) ? idn : 0;
    if (output_logprobs != nullptr) out_logprob = logf(o) - logf(sum_p);
  }
  __syncthreads();
  idn = out_id;
  if (t == 0) {
    *outputs = idn;
    if (output_logprobs != nullptr) *output_logprobs = out_logprob;
  }
}

__global__ void __launch_bounds__(BLOCKDIM_X_SAMPLE, 1)
    batch_sampling_topp_kernel(
        const int B,
        const int T,  // should be 1 typically; may not be 1 if full output is
                      // obtained
        const int V,  // vocabulary size, 60,000 ~ 120,000
        const float* __restrict__ logits,  // (B, V) if T == 1; If T != 1, only
                                           // logits[:, T-1, :] is read. This
                                           // avoids another copying operation
        int* __restrict__ outputs,         // (B,)
        float* __restrict__ output_logprobs,  // (B,)
        RAND* __restrict__ states,            // random state, typedef
                                    // curandStatePhilox4_32_10_t RAND;
        float* __restrict__ probs,  // probs (in L2 cache)
        const int* __restrict__ slot_indices,
        const float top_p,
        const int* __restrict__ sampling_status) {
  const int b = blockIdx.x;
  const int d = blockDim.x;
  const int t = threadIdx.x;
  const int w = t / 32;
  const int l = t % 32;
  if (sampling_status != nullptr &&
      (sampling_status[0] != 0 || b >= sampling_status[1])) {
    if (t == 0) {
      outputs[b] = -1;
      if (output_logprobs != nullptr) {
        output_logprobs[b] = -INFINITY;
      }
    }
    return;
  }
  __shared__ __align__(256) char reduce_buf[256];
  __builtin_assume(BLOCKDIM_X_SAMPLE == d);
  __builtin_assume(V % 4 == 0);
  __builtin_assume(V <= 1048576);
  const int V4 = V / 4;
  float4 l4, p4;

  logits += (b * T + (T - 1)) * V;                       // B T V
  outputs += b;                                          // B
  if (output_logprobs != nullptr) output_logprobs += b;  // B
  states += slot_indices == nullptr ? b : slot_indices[b];  // request slot
  probs += (b * T + (T - 1)) * V;                        // B T V

  float maxu = -INFINITY;
  for (int i = t; i < V4; i += d) {
    l4 = ((float4*)logits)[i];
#pragma unroll
    for (int j = 0; j < 4; j++) {
      float& fl = ((float*)&l4)[j];
      fl = sf(fl * float(M_LOG2E));
      maxu = max(maxu, fl);
    }
    ((float4*)probs)[i] = l4;
  }
  blockReduceAll(maxu, MaxOp<float>{}, MaxOp<float>::identity(), reduce_buf);
  __syncthreads();
  float exp_denom = 0;
  for (int i = t; i < V4; i += d) {
    l4 = ((float4*)probs)[i];
    float em = 0.f;
#pragma unroll
    for (int j = 0; j < 4; j++) {
      float& fr = ((float*)&l4)[j];
      em += exp2f(fr - maxu);
    }
    exp_denom += em;
  }
  blockReduceAll(exp_denom, SumOp<float>{}, SumOp<float>::identity(),
                 reduce_buf);
  __syncthreads();
  float pmax = -INFINITY;
  float pmin = +INFINITY;
  for (int i = t; i < V4; i += d) {
    l4 = ((float4*)probs)[i];
#pragma unroll
    for (int j = 0; j < 4; j++) {
      float& fr = ((float*)&l4)[j];
      fr = exp2f(fr - maxu) / exp_denom;
      pmax = max(pmax, fr);
      pmin = min(pmin, fr);
    }
    ((float4*)probs)[i] = l4;
  }
  blockReduceAll(pmax, MaxOp<float>{}, MaxOp<float>::identity(), reduce_buf);
  __syncthreads();
  blockReduceAll(pmin, MinOp<float>{}, MinOp<float>::identity(), reduce_buf);
  __syncthreads();

  // if(t==0) P0f(pmax);
  unsigned left = __float_as_uint(pmin), right = __float_as_uint(pmax) + 1;

  // uint4 cnt = {.x=(unsigned)V, .y=0, .z=0, .w=0};
  l4 = {.x = 1, .y = 0, .z = 0, .w = 0};
  uint4 pivot;
  while ((l4.x > top_p) && left < right - 1) {
    pivot.x = left;
    pivot.z = (left + right) / 2;
    pivot.y = (left + pivot.z) / 2;
    pivot.w = (pivot.z + right) / 2;
    l4.y = l4.z = l4.w = 0;
    for (int i = t; i < V4; i += d) {
      p4 = ((float4*)probs)[i];
#pragma unroll
      for (int j = 0; j < 4; j++) {
        float& p = ((float*)&p4)[j];
        if (p >= __uint_as_float(pivot.y)) l4.y += p;
        if (p >= __uint_as_float(pivot.z)) l4.z += p;
        if (p >= __uint_as_float(pivot.w)) l4.w += p;
      }
    }
    __syncthreads();
    blockReduceAll<float, SumOp<float>, BLOCKDIM_X_SAMPLE, true>(
        l4.y, SumOp<float>{}, SumOp<float>::identity(), reduce_buf);
    __syncthreads();
    if (l4.y < top_p) {
      left = pivot.x;
      right = pivot.y;
      continue;
    }
    blockReduceAll<float, SumOp<float>, BLOCKDIM_X_SAMPLE, true>(
        l4.z, SumOp<float>{}, SumOp<float>::identity(), reduce_buf);
    __syncthreads();
    if (l4.z < top_p) {
      left = pivot.y;
      right = pivot.z;
      l4.x = l4.y;
      continue;
    }
    blockReduceAll<float, SumOp<float>, BLOCKDIM_X_SAMPLE, true>(
        l4.w, SumOp<float>{}, SumOp<float>::identity(), reduce_buf);
    __syncthreads();
    if (l4.w < top_p) {
      left = pivot.z;
      right = pivot.w;
      l4.x = l4.z;
      continue;
    }
    left = pivot.w;
    l4.x = l4.w;
  }
  // return left
  float threshold = __uint_as_float(left);
  // if(t==0) P0f(threshold);
  // 5. recompute (read once)
  float gtp = 0;
  unsigned eqk = 0;
  __shared__ float /* seqp, */ sgtp;
  __shared__ unsigned seqk;

  for (int i = t; i < V4; i += d) {
    p4 = ((float4*)probs)[i];
#pragma unroll
    for (int j = 0; j < 4; j++) {
      float& p = ((float*)&p4)[j];
      // bool u0 = (p == threshold);
      // bool u1 = (p > threshold);
      // eqk += u0;
      // gtp = fmaf(p, u1, gtp);
      if (p == threshold) eqk++;
      if (p > threshold) gtp += p;
    }
  }
  float cgtp = blockInclusiveScan(gtp, reduce_buf, &sgtp);
  __syncthreads();
  unsigned ceqk = blockInclusiveScan(eqk, reduce_buf, &seqk);
  __syncthreads();
  unsigned neqk = seqk;
  float comp = 1.0f;
  if (neqk > 0) {
    comp = min(sf((top_p - sgtp) / (threshold * neqk)), comp);
    comp = max(comp, 0.0f);
  }

  // 6. Yield sampled tokens
  __shared__ float randp, sum_p;
  __shared__ float4 rand4;
  __shared__ int idxt;
  float actual_p = gtp + (threshold * eqk) * comp;
  __syncthreads();
  float cumu_p = blockInclusiveScan(actual_p, reduce_buf, &sum_p);
  __syncthreads();
  if (t == 0) {
    idxt = 0;
    rand4 = curand_uniform4(states);
    randp = sum_p * rand4.x;  // only once
  }
  __syncthreads();

  bool u = (randp <= cumu_p);
  // at last thread: randp = sum_p * rand4.x < cumu_p == sum_p, u == 1
  if (l == 31) ((unsigned*)reduce_buf)[w] = u;
  __syncthreads();
  bool u_ = __shfl_up_sync(0xffffffff, u, 1);
  if (t == 0)
    u_ = 0;
  else if (l == 0)
    u_ = ((unsigned*)reduce_buf)[w - 1];
  __syncthreads();

  if (u != u_) idxt = t;
  __syncthreads();

  // a sub-tile (of no more than 1024)
  int idn = idxt * 4 + (t / 4) * 4 * d + (t % 4);
  // .... .... (idxt) |||| .... .... .... |||| .... .... .... |||| ....
  float o0 = (idn < V) ? (probs[idn]) : 0;
  float o = (o0 < threshold) ? 0 : (o0 == threshold) ? (o0 * comp) : o0;

  __shared__ float sum_o;
  float cumu_o = blockInclusiveScan(o, reduce_buf, &sum_o);  // monotone
  __syncthreads();
  float rand_2 = sum_o * rand4.y;
  u = (rand_2 <= cumu_o);
  // at last thread: cumu_o == sum_o, rand4.y < 1, sum_o * rand4.y < cumu_o, u
  // == 1
  if (l == 31) ((unsigned*)reduce_buf)[w] = u;
  // u: current u_: prev
  // at first thread: u_ == 0
  u_ = __shfl_up_sync(0xffffffff, u, 1);
  __syncthreads();
  if (t == 0)
    u_ = 0;
  else if (l == 0)
    u_ = ((unsigned*)reduce_buf)[w - 1];
  __syncthreads();

  // write idn
  __shared__ int out_id;
  __shared__ float out_logprob;
  if (u != u_) {
    out_id = (idn < V) ? idn : 0;
    if (output_logprobs != nullptr) out_logprob = logf(o) - logf(sum_p);
  }
  __syncthreads();
  idn = out_id;
  if (t == 0) {
    *outputs = idn;
    if (output_logprobs != nullptr) *output_logprobs = out_logprob;
  }
}

std::vector<at::Tensor> batch_sampling_temperature_topk_topp(
    at::Tensor& logits, at::Tensor& states, double temperature, int64_t top_k,
    double top_p, bool return_logprobs) {
  int B, T, V;
  if (logits.dtype() != at::kFloat) {
    throw std::invalid_argument(
        "Logits tensor must be of type float32 (FP32), got " +
        std::string(logits.dtype().name()) + " !\n");
  }
  V = logits.size(-1);
  B = (logits.dim() >= 2) ? logits.size(0) : 1;
  T = (logits.dim() == 3) ? logits.size(1) : 1;

  if (!(V > 0 && V <= 1048576 && V % 4 == 0)) {
    throw std::invalid_argument(
        "Vocabulary size must be multiple of 4, and no larger than 1048576, "
        "got " +
        std::to_string(V) + " !\n");
  }
  if (!(B > 0 && T > 0)) {
    throw std::invalid_argument(
        "B and T must be positive, got B=" + std::to_string(B) +
        ", T=" + std::to_string(T) + " !\n");
  }
  if (!(temperature >= 0.001 && temperature <= 1000)) {
    throw std::invalid_argument("Temperature outside range, got " +
                                std::to_string(temperature) +
                                ", expect [0.001, 1000]!\n");
  }
  if (top_k <= 0 || top_k > V) top_k = V;
  if (top_p < 0 || top_p > 1) top_p = 1;
  if (top_p == 0) {
    top_k = 1;
    top_p = 1;
  }
  double log2e_inv_temp = M_LOG2E / temperature;

  auto stream = at::cuda::getCurrentCUDAStream();
  auto probs = at::empty(
      {B, V}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));
  if (B * V * 4 <= 4194304) {
    cudaStreamAttrValue stream_attribute;
    stream_attribute.accessPolicyWindow.base_ptr = probs.data_ptr();
    stream_attribute.accessPolicyWindow.num_bytes = B * V * 4;
    stream_attribute.accessPolicyWindow.hitRatio = 1;
    stream_attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    stream_attribute.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
    cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow,
                           &stream_attribute);
  }
  auto out =
      at::empty({B}, at::TensorOptions().dtype(at::kInt).device(at::kCUDA));
  std::vector<at::Tensor> outputs = {out};
  float* output_logprobs = nullptr;
  if (return_logprobs) {
    auto out_logprobs =
        at::empty({B}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));
    output_logprobs = (float*)out_logprobs.data_ptr();
    outputs.push_back(out_logprobs);
  }
  if (temperature == 1 && top_k == V)
    batch_sampling_topp_kernel<<<B, 1024, 0, stream>>>(
        B, T, V, (float*)logits.data_ptr(), (int*)out.data_ptr(),
        output_logprobs, (RAND*)states.data_ptr(), (float*)probs.data_ptr(),
        nullptr, (float)top_p, nullptr);
  else
    batch_sampling_temperature_topk_topp_kernel<<<B, 1024, 0, stream>>>(
        B, T, V, (float*)logits.data_ptr(), (int*)out.data_ptr(),
        output_logprobs, (RAND*)states.data_ptr(), (float*)probs.data_ptr(),
        nullptr, (float)log2e_inv_temp, (int)top_k, (float)top_p, nullptr);
  return outputs;
}

namespace {

at::Tensor sampling_probs_workspace(const at::Tensor& logits) {
  const int64_t bytes = logits.numel() * static_cast<int64_t>(sizeof(float));
  auto probs = at::empty(logits.sizes(), logits.options().dtype(at::kFloat));
  if (bytes <= 4194304) {
    auto stream = at::cuda::getCurrentCUDAStream();
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    C10_CUDA_CHECK(cudaStreamIsCapturing(stream, &capture_status));
    if (capture_status == cudaStreamCaptureStatusNone) {
      cudaStreamAttrValue attribute{};
      attribute.accessPolicyWindow.base_ptr = probs.data_ptr();
      attribute.accessPolicyWindow.num_bytes = static_cast<size_t>(bytes);
      attribute.accessPolicyWindow.hitRatio = 1.0f;
      attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
      attribute.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
      C10_CUDA_CHECK(cudaStreamSetAttribute(
          stream, cudaStreamAttributeAccessPolicyWindow, &attribute));
    }
  }
  return probs;
}

at::Tensor sampling_output(const at::Tensor& logits) {
  return at::empty({logits.size(0)}, logits.options().dtype(at::kInt));
}

struct SamplingMetadata {
  at::Tensor status;
  at::Tensor workspace;
};

SamplingMetadata prepare_sampling_metadata(
    const at::Tensor& logits,
    const at::Tensor& states,
    const at::Tensor& slot_indices,
    const at::Tensor& num_active_samples) {
  auto status = at::empty({2}, slot_indices.options());
  auto workspace = at::empty({states.size(0)}, slot_indices.options());
  validate_sampling_metadata_kernel<<<
      1, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      slot_indices.data_ptr<int>(),
      num_active_samples.defined()
          ? num_active_samples.data_ptr<int>()
          : nullptr,
      static_cast<int>(logits.size(0)), static_cast<int>(states.size(0)),
      status.data_ptr<int>(), workspace.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return SamplingMetadata{std::move(status), std::move(workspace)};
}

void normalize_sampling_controls(
    int vocab_size, double& temperature, int64_t& top_k, double& top_p) {
  temperature = fmin(fmax(temperature, 0.001), 1000.0);
  if (top_k <= 0 || top_k > vocab_size) top_k = vocab_size;
  top_p = fmin(fmax(top_p, 0.0), 1.0);
  if (top_p == 0.0) {
    top_k = 1;
    top_p = 1.0;
  }
}

}  // namespace

at::Tensor sampling_temperature_topk_topp_scalar_cuda(
    at::Tensor logits, at::Tensor states, at::Tensor slot_indices,
    double temperature, int64_t top_k, double top_p,
    at::Tensor num_active_samples) {
  const int B = static_cast<int>(logits.size(0));
  const int V = static_cast<int>(logits.size(1));
  normalize_sampling_controls(V, temperature, top_k, top_p);
  auto probs = sampling_probs_workspace(logits);
  auto output = sampling_output(logits);
  auto metadata = prepare_sampling_metadata(
      logits, states, slot_indices, num_active_samples);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (temperature == 1.0 && top_k == V) {
    batch_sampling_topp_kernel<<<B, BLOCKDIM_X_SAMPLE, 0, stream>>>(
        B, 1, V, logits.data_ptr<float>(), output.data_ptr<int>(), nullptr,
        reinterpret_cast<RAND*>(states.data_ptr()), probs.data_ptr<float>(),
        slot_indices.data_ptr<int>(), static_cast<float>(top_p),
        metadata.status.data_ptr<int>());
  } else {
    batch_sampling_temperature_topk_topp_kernel
        <<<B, BLOCKDIM_X_SAMPLE, 0, stream>>>(
            B, 1, V, logits.data_ptr<float>(), output.data_ptr<int>(), nullptr,
            reinterpret_cast<RAND*>(states.data_ptr()), probs.data_ptr<float>(),
            slot_indices.data_ptr<int>(),
            static_cast<float>(M_LOG2E / temperature),
            static_cast<int>(top_k), static_cast<float>(top_p),
            metadata.status.data_ptr<int>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor sampling_temperature_topk_topp_per_request_cuda(
    at::Tensor logits, at::Tensor states, at::Tensor slot_indices,
    at::Tensor temperatures, at::Tensor top_ks, at::Tensor top_ps,
    at::Tensor num_active_samples) {
  const int B = static_cast<int>(logits.size(0));
  const int V = static_cast<int>(logits.size(1));
  auto probs = sampling_probs_workspace(logits);
  auto output = sampling_output(logits);
  auto metadata = prepare_sampling_metadata(
      logits, states, slot_indices, num_active_samples);
  auto stream = at::cuda::getCurrentCUDAStream();
  batch_sampling_repetition_temperature_topk_topp_kernel
      <<<B, BLOCKDIM_X_SAMPLE, 0, stream>>>(
          B, 1, V, logits.data_ptr<float>(), nullptr,
          slot_indices.data_ptr<int>(), output.data_ptr<int>(), nullptr,
          reinterpret_cast<RAND*>(states.data_ptr()), probs.data_ptr<float>(),
          nullptr, nullptr, nullptr, temperatures.data_ptr<float>(),
          top_ks.data_ptr<int>(), top_ps.data_ptr<float>(), 0.0f, 0.0f, 1.0f,
          1.0f, V, 1.0f, metadata.status.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor sampling_six_parameter_scalar_cuda(
    at::Tensor logits, at::Tensor penalties, at::Tensor states,
    at::Tensor slot_indices, double presence_penalty, double frequency_penalty,
    double penalty_decay, double temperature, int64_t top_k, double top_p,
    at::Tensor num_active_samples) {
  const int B = static_cast<int>(logits.size(0));
  const int V = static_cast<int>(logits.size(1));
  normalize_sampling_controls(V, temperature, top_k, top_p);
  auto probs = sampling_probs_workspace(logits);
  auto output = sampling_output(logits);
  auto metadata = prepare_sampling_metadata(
      logits, states, slot_indices, num_active_samples);
  auto stream = at::cuda::getCurrentCUDAStream();
  batch_sampling_repetition_temperature_topk_topp_kernel
      <<<B, BLOCKDIM_X_SAMPLE, 0, stream>>>(
          B, 1, V, logits.data_ptr<float>(), penalties.data_ptr<float>(),
          slot_indices.data_ptr<int>(), output.data_ptr<int>(), nullptr,
          reinterpret_cast<RAND*>(states.data_ptr()), probs.data_ptr<float>(),
          nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
          static_cast<float>(presence_penalty),
          static_cast<float>(frequency_penalty),
          static_cast<float>(penalty_decay), static_cast<float>(temperature),
          static_cast<int>(top_k), static_cast<float>(top_p),
          metadata.status.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor sampling_six_parameter_per_request_cuda(
    at::Tensor logits, at::Tensor penalties, at::Tensor states,
    at::Tensor slot_indices, at::Tensor presence_penalties,
    at::Tensor frequency_penalties, at::Tensor penalty_decays,
    at::Tensor temperatures, at::Tensor top_ks, at::Tensor top_ps,
    at::Tensor num_active_samples) {
  const int B = static_cast<int>(logits.size(0));
  const int V = static_cast<int>(logits.size(1));
  auto probs = sampling_probs_workspace(logits);
  auto output = sampling_output(logits);
  auto metadata = prepare_sampling_metadata(
      logits, states, slot_indices, num_active_samples);
  auto stream = at::cuda::getCurrentCUDAStream();
  batch_sampling_repetition_temperature_topk_topp_kernel
      <<<B, BLOCKDIM_X_SAMPLE, 0, stream>>>(
          B, 1, V, logits.data_ptr<float>(), penalties.data_ptr<float>(),
          slot_indices.data_ptr<int>(), output.data_ptr<int>(), nullptr,
          reinterpret_cast<RAND*>(states.data_ptr()), probs.data_ptr<float>(),
          presence_penalties.data_ptr<float>(),
          frequency_penalties.data_ptr<float>(), penalty_decays.data_ptr<float>(),
          temperatures.data_ptr<float>(), top_ks.data_ptr<int>(),
          top_ps.data_ptr<float>(), 0.0f, 0.0f, 1.0f, 1.0f, V, 1.0f,
          metadata.status.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
