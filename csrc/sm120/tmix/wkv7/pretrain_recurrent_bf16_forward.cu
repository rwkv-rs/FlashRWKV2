// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to BlinkDL/RWKV-LM
// Canonical source: RWKV-LM RWKV-v7/train_temp/cuda/rwkv7_clampw_v3_for_h100.cu
// Source revision: 952102498e9ed367ea0a59ee64106916d474d30f.
// Local adaptation:
//   - D=64 preserves the train_temp clampw-v3 implementation;
//   - D=128 follows rwkv7_clampw128_v2's 64-wide segmented recurrence;
//   - D=256 is a local warp-tiled extension of the same math and layout.

#include <assert.h>
#include <cuda_runtime.h>

#ifdef _FP32_
    using bf = float;
    #define to_float(u) (u)
    #define to_bf(u) (u)
#else
    #include <cuda_bf16.h>
    using bf = __nv_bfloat16;
    #define to_float(u) (__bfloat162float(u))
    #define to_bf(u) (__float2bfloat16_rn(u))
#endif

using i64 = long long int;
typedef bf * __restrict__ F_;
constexpr float W_SCALE = -0.6065306597f; // -exp(-0.5)

// Fast kernel for H100 etc. (benchmark fwd & bwd speed if you are using consumer GPUs)

//######################################################################################################

template<int N> __launch_bounds__(N,2)
__global__ void forward_kernel_preload(int T,int H,F_ r_,F_ w_,F_ k_,F_ v_,F_ a_,F_ b_,bf* __restrict__ y_,float* s__,float* __restrict__ sa_)
{
    const int bb=blockIdx.y, hh=blockIdx.x, i=threadIdx.x;
    float* __restrict__ s_ = s__ + i64(bb*H+hh) * i64((T/_CHUNK_LEN_)*N*N);
    float state[N];
#pragma unroll
    for (int j=0; j<N; ++j) {
        state[j] = 0.0f;
    }
    __shared__ float r[_CHUNK_LEN_][N];
    __shared__ float w[_CHUNK_LEN_][N];
    __shared__ float k[_CHUNK_LEN_][N];
    __shared__ float v[_CHUNK_LEN_][N];
    __shared__ float a[_CHUNK_LEN_][N];
    __shared__ float b[_CHUNK_LEN_][N];

    for (int t0 = 0; t0 < T; t0 += _CHUNK_LEN_)
    {
        __syncthreads();
#pragma unroll
        for (int tt=0; tt<_CHUNK_LEN_; ++tt) {
            const int idx = ((bb*T+t0+tt)*H+hh)*N+i;
            r[tt][i] = to_float(r_[idx]);
            w[tt][i] = __expf(W_SCALE / (1.0f + __expf(-to_float(w_[idx]))));
            k[tt][i] = to_float(k_[idx]);
            v[tt][i] = to_float(v_[idx]);
            a[tt][i] = to_float(a_[idx]);
            b[tt][i] = to_float(b_[idx]);
        }
        __syncthreads();

        for (int tt=0; tt<_CHUNK_LEN_; ++tt) {
            const int idx = ((bb*T+t0+tt)*H+hh)*N+i;

            float sa = 0.0f;
#pragma unroll
            for (int j=0; j<N; ++j) {
                sa += state[j] * a[tt][j];
            }
            sa_[idx] = sa;

            float vi = v[tt][i];
            float y=0.0f;
#pragma unroll
            for (int j=0; j<N; ++j) {
                float s = state[j];
                s = s * w[tt][j] + (sa * b[tt][j] + k[tt][j] * vi);
                y += s * r[tt][j];
                state[j] = s;
            }

            y_[idx] = to_bf(y);
        }

        {
            int base = (t0/_CHUNK_LEN_)*N*N + i;
#pragma unroll
            for (int j=0; j<N; ++j) {
                s_[base+j*N] = state[j];
            }
        }
        // Safe without a tail barrier: the next chunk starts with __syncthreads()
        // before any thread overwrites the shared preload buffers.
    }
}
void cuda_forward_v3(int B,int T,int H,bf*r,bf*w,bf*k,bf*v,bf*a,bf*b,bf*y,float*s,float*sa)
{
    forward_kernel_preload<_N_><<<dim3(H,B),dim3(_N_)>>>(T,H,r,w,k,v,a,b,y,s,sa);
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset=16; offset>0; offset>>=1)
        value += __shfl_down_sync(0xffffffffu, value, offset);
    return value;
}

template<int N> __launch_bounds__(32,4)
__global__ void forward_kernel_split64(int T,int H,F_ r_,F_ w_,F_ k_,F_ v_,F_ a_,F_ b_,bf* __restrict__ y_,float* s__,float* __restrict__ sa_)
{
    constexpr int KEYS_PER_LANE=N/32;
    static_assert(N==128 || N==256);
    const int value=blockIdx.x, hh=blockIdx.y, bb=blockIdx.z, lane=threadIdx.x;
    float state[KEYS_PER_LANE]={0};
    float* __restrict__ s_=s__+i64(bb*H+hh)*i64((T/_CHUNK_LEN_)*N*N);
    for(int t=0;t<T;++t) {
        const int64_t token_base=(i64(bb*T+t)*H+hh)*N;
        float state_dot_a=0.0f;
#pragma unroll
        for(int q=0;q<KEYS_PER_LANE;++q) {
            const int key=lane+q*32;
            state_dot_a=fmaf(state[q],to_float(a_[token_base+key]),state_dot_a);
        }
        state_dot_a=warp_sum(state_dot_a);
        state_dot_a=__shfl_sync(0xffffffffu,state_dot_a,0);
        if(lane==0) sa_[token_base+value]=state_dot_a;
        const float value_token=to_float(v_[token_base+value]);
        float output=0.0f;
#pragma unroll
        for(int q=0;q<KEYS_PER_LANE;++q) {
            const int key=lane+q*32;
            const float retention=__expf(W_SCALE/(1.0f+__expf(-to_float(w_[token_base+key]))));
            state[q]=state[q]*retention+state_dot_a*to_float(b_[token_base+key])+
                     to_float(k_[token_base+key])*value_token;
            output=fmaf(state[q],to_float(r_[token_base+key]),output);
        }
        output=warp_sum(output);
        if(lane==0) y_[token_base+value]=to_bf(output);
        if((t+1)%_CHUNK_LEN_==0) {
            const int64_t chunk_base=i64(t/_CHUNK_LEN_)*N*N;
#pragma unroll
            for(int q=0;q<KEYS_PER_LANE;++q) {
                const int key=lane+q*32;
                s_[chunk_base+i64(key)*N+value]=state[q];
            }
        }
    }
}

void cuda_forward_split(int B,int T,int H,int N,bf*r,bf*w,bf*k,bf*v,bf*a,bf*b,bf*y,float*s,float*sa)
{
    if (N == 128) {
        forward_kernel_split64<128><<<dim3(128,H,B),32>>>(T,H,r,w,k,v,a,b,y,s,sa);
    } else {
        forward_kernel_split64<256><<<dim3(256,H,B),32>>>(T,H,r,w,k,v,a,b,y,s,sa);
    }
}

//######################################################################################################

template<int N, int TILE>
__global__ void backward_kernel_preload(int T, int H, F_ r_, F_ w_, F_ k_, F_ v_, F_ a_, F_ b_, F_ dy_, float * __restrict__ s__, float * __restrict__ sa_, bf* dr_, bf* dw_, bf* dk_, bf* dv_, bf* da_, bf* db_)
{
    int bb = blockIdx.y, hh = blockIdx.x, i = threadIdx.x;
    float* __restrict__ s_ = s__ + i64(bb*H+hh) * i64((T/_CHUNK_LEN_)*N*N);

    float stateT[N] = {0}, dstate[N] = {0}, dstateT[N] = {0};
    static_assert(_CHUNK_LEN_%TILE == 0, "TILE must divide _CHUNK_LEN_");
    __shared__ float r[TILE][N];
    __shared__ float w[TILE][N];
    __shared__ float ws[TILE][N];
    __shared__ float k[TILE][N];
    __shared__ float v[TILE][N];
    __shared__ float a[TILE][N];
    __shared__ float b[TILE][N];
    __shared__ float dy[TILE][N];
    __shared__ float sa[TILE][N];
    __shared__ float dSb_shared[N];
    float ri, wi, ki, ai, bi, dyi;

    for (int t0 = T-_CHUNK_LEN_; t0 >= 0; t0 -= _CHUNK_LEN_)
    {
        {
            int base = (t0/_CHUNK_LEN_)*N*N + i*N;
            const float4* s4 = (const float4*)(s_ + base);
#pragma unroll
            for (int j4 = 0; j4 < N/4; j4++) {
                float4 q = s4[j4];
                const int j = j4<<2;
                stateT[j+0] = q.x;
                stateT[j+1] = q.y;
                stateT[j+2] = q.z;
                stateT[j+3] = q.w;
            }
        }

        for (int subt=_CHUNK_LEN_-TILE; subt>=0; subt-=TILE) {
            __syncthreads();
#pragma unroll
            for (int tt=0; tt<TILE; ++tt) {
                int idx = bb*T*H*N + (t0+subt+tt)*H*N + hh * N + i;
                r[tt][i] = to_float(r_[idx]);
                float w_sig = 1.0f / (1.0f + __expf(-to_float(w_[idx])));
                float wi = __expf(W_SCALE * w_sig);
                if constexpr (TILE == 8) {
                    ws[tt][i] = W_SCALE * wi * w_sig * (1.0f - w_sig);
                } else {
                    ws[tt][i] = w_sig;
                }
                w[tt][i] = wi;
                k[tt][i] = to_float(k_[idx]);
                v[tt][i] = to_float(v_[idx]);
                a[tt][i] = to_float(a_[idx]);
                b[tt][i] = to_float(b_[idx]);
                dy[tt][i] = to_float(dy_[idx]);
                sa[tt][i] = sa_[idx];
            }
            __syncthreads();

            for (int tt=TILE-1; tt>=0; --tt) {
                int idx = bb*T*H*N + (t0+subt+tt)*H*N + hh * N + i;
                ri = r[tt][i];
                wi = w[tt][i];
                ki = k[tt][i];
                ai = a[tt][i];
                bi = b[tt][i];
                dyi = dy[tt][i];

                float dr = 0;
#pragma unroll
                for (int j = 0; j < N; j++) {
                    dr += stateT[j] * dy[tt][j];
                }
                dr_[idx] = to_bf(dr);

                float iwi = 1.0f / wi;
#pragma unroll
                for (int j = 0; j < N; j++) {
                    stateT[j] = (stateT[j] - ki * v[tt][j] - bi * sa[tt][j]) * iwi;
                    dstate[j] += dyi * r[tt][j];
                    dstateT[j] += ri * dy[tt][j];
                }

                float dw = 0, dk = 0, dv = 0, db = 0, dSb = 0;
#pragma unroll
                for (int j = 0; j < N; j++) {
                    dw += dstateT[j] * stateT[j];
                    dk += dstateT[j] * v[tt][j];
                    dv += dstate[j] * k[tt][j];
                    dSb += dstate[j] * b[tt][j];
                    db += dstateT[j] * sa[tt][j];
                }
                if constexpr (TILE == 8) {
                    dw_[idx] = to_bf(dw * ws[tt][i]);
                } else {
                    float w_sig = ws[tt][i];
                    dw_[idx] = to_bf(W_SCALE * dw * wi * w_sig * (1.0f - w_sig));
                }

                dk_[idx] = to_bf(dk);
                dv_[idx] = to_bf(dv);
                db_[idx] = to_bf(db);

                __syncthreads();
                dSb_shared[i] = dSb;
                __syncthreads();

                float da = 0;
#pragma unroll
                for (int j = 0; j < N; j++) {
                    da += stateT[j]*dSb_shared[j];
                }
                da_[idx] = to_bf(da);

#pragma unroll
                for (int j = 0; j < N; j++) {
                    dstate[j] = dstate[j] * w[tt][j] + dSb * a[tt][j];
                    dstateT[j] = dstateT[j] * wi + ai * dSb_shared[j];
                }
            }
        }
    }
}

void cuda_backward_v3(int B, int T, int H, bf*r, bf*w, bf*k, bf*v, bf*a, bf*b, bf*dy, float*s, float*sa, bf*dr, bf*dw, bf*dk, bf*dv, bf*da, bf*db)
{
    assert(T%_CHUNK_LEN_ == 0);
    backward_kernel_preload<_N_,16><<<dim3(H,B), dim3(_N_)>>>(T,H,r,w,k,v,a,b,dy,s,sa,dr,dw,dk,dv,da,db);
}

template<int N, int SEGMENTS> __launch_bounds__(N * SEGMENTS, 1)
__global__ void backward_kernel_split64(int T,int H,F_ r_,F_ w_,F_ k_,F_ v_,F_ a_,F_ b_,F_ dy_,float* __restrict__ s__,float* __restrict__ sa_,bf* dr_,bf* dw_,bf* dk_,bf* dv_,bf* da_,bf* db_)
{
    constexpr int SEG=64, TILE=4;
    static_assert(N == SEG * SEGMENTS);
    const int bb=blockIdx.y, hh=blockIdx.x, i=threadIdx.x, seg=threadIdx.y;
    const int j0=seg*SEG;
    float* __restrict__ s_=s__+i64(bb*H+hh)*i64((T/_CHUNK_LEN_)*N*N);
    float stateT[SEG]={0}, dstate[SEG]={0}, dstateT[SEG]={0};
    __shared__ float r[TILE][N],w[TILE][N],ws[TILE][N],k[TILE][N];
    __shared__ float v[TILE][N],a[TILE][N],b[TILE][N],dy[TILE][N],sa[TILE][N];
    __shared__ float dSb_shared[N], partial[SEGMENTS][N];

    for (int t0=T-_CHUNK_LEN_; t0>=0; t0-=_CHUNK_LEN_) {
        const int base=(t0/_CHUNK_LEN_)*N*N+i*N+j0;
        const float4* s4=(const float4*)(s_+base);
#pragma unroll
        for (int j4=0; j4<SEG/4; ++j4) {
            const float4 q=s4[j4]; const int j=j4*4;
            stateT[j]=q.x; stateT[j+1]=q.y; stateT[j+2]=q.z; stateT[j+3]=q.w;
        }
        for (int subt=_CHUNK_LEN_-TILE; subt>=0; subt-=TILE) {
            __syncthreads();
            if (seg==0) {
#pragma unroll
                for (int tt=0; tt<TILE; ++tt) {
                    const int idx=((bb*T+t0+subt+tt)*H+hh)*N+i;
                    r[tt][i]=to_float(r_[idx]);
                    const float sig=1.0f/(1.0f+__expf(-to_float(w_[idx])));
                    const float wi=__expf(W_SCALE*sig);
                    w[tt][i]=wi; ws[tt][i]=W_SCALE*wi*sig*(1.0f-sig);
                    k[tt][i]=to_float(k_[idx]); v[tt][i]=to_float(v_[idx]);
                    a[tt][i]=to_float(a_[idx]); b[tt][i]=to_float(b_[idx]);
                    dy[tt][i]=to_float(dy_[idx]); sa[tt][i]=sa_[idx];
                }
            }
            __syncthreads();
            for (int tt=TILE-1; tt>=0; --tt) {
                const int idx=((bb*T+t0+subt+tt)*H+hh)*N+i;
                const float ri=r[tt][i], wi=w[tt][i], ki=k[tt][i];
                const float ai=a[tt][i], bi=b[tt][i], dyi=dy[tt][i];
                float dr=0.0f;
#pragma unroll
                for (int j=0; j<SEG; ++j) dr += stateT[j]*dy[tt][j0+j];
                partial[seg][i]=dr; __syncthreads();
                if (seg==0) { float q=0; for(int z=0;z<SEGMENTS;++z) q+=partial[z][i]; dr_[idx]=to_bf(q); }
                __syncthreads();
                const float iwi=1.0f/wi;
#pragma unroll
                for (int j=0; j<SEG; ++j) {
                    const int jj=j0+j;
                    stateT[j]=(stateT[j]-ki*v[tt][jj]-bi*sa[tt][jj])*iwi;
                    dstate[j]+=dyi*r[tt][jj]; dstateT[j]+=ri*dy[tt][jj];
                }
                float dw=0,dk=0,dv=0,db=0,dSb=0;
#pragma unroll
                for (int j=0; j<SEG; ++j) {
                    const int jj=j0+j;
                    dw+=dstateT[j]*stateT[j]; dk+=dstateT[j]*v[tt][jj];
                    dv+=dstate[j]*k[tt][jj]; dSb+=dstate[j]*b[tt][jj];
                    db+=dstateT[j]*sa[tt][jj];
                }
#define REDUCE_SEGMENT(value, target) \
                partial[seg][i]=(value); __syncthreads(); \
                if(seg==0){float q=0; for(int z=0;z<SEGMENTS;++z)q+=partial[z][i]; target;} \
                __syncthreads()
                REDUCE_SEGMENT(dw, dw_[idx]=to_bf(q*ws[tt][i]));
                REDUCE_SEGMENT(dk, dk_[idx]=to_bf(q));
                REDUCE_SEGMENT(dv, dv_[idx]=to_bf(q));
                REDUCE_SEGMENT(db, db_[idx]=to_bf(q));
                REDUCE_SEGMENT(dSb, dSb_shared[i]=q);
#undef REDUCE_SEGMENT
                float da=0.0f;
#pragma unroll
                for (int j=0; j<SEG; ++j) da+=stateT[j]*dSb_shared[j0+j];
                partial[seg][i]=da; __syncthreads();
                if(seg==0){float q=0;for(int z=0;z<SEGMENTS;++z)q+=partial[z][i];da_[idx]=to_bf(q);}
                __syncthreads();
                const float dSb_i=dSb_shared[i];
#pragma unroll
                for (int j=0; j<SEG; ++j) {
                    const int jj=j0+j;
                    dstate[j]=dstate[j]*w[tt][jj]+dSb_i*a[tt][jj];
                    dstateT[j]=dstateT[j]*wi+ai*dSb_shared[jj];
                }
            }
        }
    }
}

void cuda_backward_split(int B,int T,int H,int N,bf*r,bf*w,bf*k,bf*v,bf*a,bf*b,bf*dy,float*s,float*sa,bf*dr,bf*dw,bf*dk,bf*dv,bf*da,bf*db)
{
    assert(T%_CHUNK_LEN_==0);
    assert(N==128);
    backward_kernel_split64<128,2><<<dim3(H,B),dim3(128,2)>>>(T,H,r,w,k,v,a,b,dy,s,sa,dr,dw,dk,dv,da,db);
}

// D256 local backward specialization. The value-column recurrence that
// produces dv/dSb is independent from the key-row recurrence that produces
// dr/dw/dk/db/da once dSb is materialized. Splitting those two scans keeps
// only 8 or 16 floats per lane instead of three float[64] arrays in a
// 1024-thread CTA, and creates D warps per head for enough SM120 parallelism.
template<int N> __launch_bounds__(32,4)
__global__ void backward_value_columns(
    int T,int H,F_ r_,F_ w_,F_ k_,F_ b_,F_ a_,F_ dy_,bf* dv_,float* dsb_)
{
    constexpr int ITEMS=N/32;
    const int value=blockIdx.x,hh=blockIdx.y,bb=blockIdx.z,lane=threadIdx.x;
    float dstate[ITEMS]={0};
    for(int t=T-1;t>=0;--t) {
        const i64 base=(i64(bb*T+t)*H+hh)*N;
        const float dyi=to_float(dy_[base+value]);
        float dv=0.0f,dSb=0.0f;
#pragma unroll
        for(int q=0;q<ITEMS;++q) {
            const int key=lane+q*32;
            dstate[q]=fmaf(dyi,to_float(r_[base+key]),dstate[q]);
            dv=fmaf(dstate[q],to_float(k_[base+key]),dv);
            dSb=fmaf(dstate[q],to_float(b_[base+key]),dSb);
        }
        dv=warp_sum(dv); dSb=warp_sum(dSb);
        dSb=__shfl_sync(0xffffffffu,dSb,0);
        if(lane==0) {
            dv_[base+value]=to_bf(dv);
            dsb_[base+value]=dSb;
        }
#pragma unroll
        for(int q=0;q<ITEMS;++q) {
            const int key=lane+q*32;
            const float sig=1.0f/(1.0f+__expf(-to_float(w_[base+key])));
            const float wi=__expf(W_SCALE*sig);
            dstate[q]=fmaf(dSb,to_float(a_[base+key]),dstate[q]*wi);
        }
    }
}

template<int N> __launch_bounds__(32,4)
__global__ void backward_key_rows(
    int T,int H,F_ r_,F_ w_,F_ k_,F_ v_,F_ a_,F_ b_,F_ dy_,
    float* s__,float* sa_,float* dsb_,bf* dr_,bf* dw_,bf* dk_,bf* da_,bf* db_)
{
    constexpr int ITEMS=N/32;
    const int key=blockIdx.x,hh=blockIdx.y,bb=blockIdx.z,lane=threadIdx.x;
    float dstateT[ITEMS]={0};
    float* s_=s__+i64(bb*H+hh)*i64((T/_CHUNK_LEN_)*N*N);
    for(int t0=T-_CHUNK_LEN_;t0>=0;t0-=_CHUNK_LEN_) {
        float stateT[ITEMS];
        const i64 chunk_base=i64(t0/_CHUNK_LEN_)*N*N+i64(key)*N;
#pragma unroll
        for(int q=0;q<ITEMS;++q) stateT[q]=s_[chunk_base+lane+q*32];
        for(int tt=_CHUNK_LEN_-1;tt>=0;--tt) {
            const int t=t0+tt; const i64 base=(i64(bb*T+t)*H+hh)*N;
            const float ri=to_float(r_[base+key]);
            const float sig=1.0f/(1.0f+__expf(-to_float(w_[base+key])));
            const float wi=__expf(W_SCALE*sig),iwi=1.0f/wi;
            const float ki=to_float(k_[base+key]);
            const float ai=to_float(a_[base+key]);
            const float bi=to_float(b_[base+key]);
            float dr=0.0f,dw=0.0f,dk=0.0f,db=0.0f,da=0.0f;
#pragma unroll
            for(int q=0;q<ITEMS;++q) {
                const int value=lane+q*32;
                dr=fmaf(stateT[q],to_float(dy_[base+value]),dr);
                stateT[q]=(stateT[q]-ki*to_float(v_[base+value])-bi*sa_[base+value])*iwi;
                dstateT[q]=fmaf(ri,to_float(dy_[base+value]),dstateT[q]);
                dw=fmaf(dstateT[q],stateT[q],dw);
                dk=fmaf(dstateT[q],to_float(v_[base+value]),dk);
                db=fmaf(dstateT[q],sa_[base+value],db);
                da=fmaf(stateT[q],dsb_[base+value],da);
            }
            dr=warp_sum(dr); dw=warp_sum(dw); dk=warp_sum(dk);
            db=warp_sum(db); da=warp_sum(da);
            if(lane==0) {
                dr_[base+key]=to_bf(dr);
                dw_[base+key]=to_bf(dw*W_SCALE*wi*sig*(1.0f-sig));
                dk_[base+key]=to_bf(dk); db_[base+key]=to_bf(db);
                da_[base+key]=to_bf(da);
            }
#pragma unroll
            for(int q=0;q<ITEMS;++q) {
                const int value=lane+q*32;
                dstateT[q]=fmaf(ai,dsb_[base+value],dstateT[q]*wi);
            }
        }
    }
}

void cuda_backward_tiled256(int B,int T,int H,bf*r,bf*w,bf*k,bf*v,bf*a,bf*b,bf*dy,float*s,float*sa,float*dsb,bf*dr,bf*dw,bf*dk,bf*dv,bf*da,bf*db)
{
    assert(T%_CHUNK_LEN_==0);
    const dim3 grid(256,H,B);
    backward_value_columns<256><<<grid,32>>>(T,H,r,w,k,b,a,dy,dv,dsb);
    backward_key_rows<256><<<grid,32>>>(T,H,r,w,k,v,a,b,dy,s,sa,dsb,dr,dw,dk,da,db);
}
