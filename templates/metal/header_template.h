#ifndef CUDAMetalKernel_h
#define CUDAMetalKernel_h

#include <metal_stdlib>
#include <metal_atomic>
#include <metal_simdgroup>
#include <metal_math>

using namespace metal;

// CUDA-style vector types
struct int2 { int x, y; };
struct int3 { int x, y, z; };
struct int4 { int x, y, z, w; };
struct uint2 { uint x, y; };
struct uint3 { uint x, y, z; };
struct uint4 { uint x, y, z, w; };
struct float2 { float x, y; };
struct float3 { float x, y, z; };
struct float4 { float x, y, z, w; };

// Thread indexing
#define threadIdx_x (thread_position_in_threadgroup.x)
#define threadIdx_y (thread_position_in_threadgroup.y)
#define threadIdx_z (thread_position_in_threadgroup.z)
#define blockIdx_x (threadgroup_position_in_grid.x)
#define blockIdx_y (threadgroup_position_in_grid.y)
#define blockIdx_z (threadgroup_position_in_grid.z)
#define blockDim_x (threads_per_threadgroup.x)
#define blockDim_y (threads_per_threadgroup.y)
#define blockDim_z (threads_per_threadgroup.z)
#define gridDim_x (threadgroups_per_grid.x)
#define gridDim_y (threadgroups_per_grid.y)
#define gridDim_z (threadgroups_per_grid.z)

// Common kernel parameters structure
struct KernelParameters {
    uint problemSize;
    uint batchSize;
    float learningRate;
    float4 reserved;  // For alignment
};

// CUDA synchronization primitives
#define __syncthreads() threadgroup_barrier(mem_flags::mem_threadgroup)
#define __threadfence() threadgroup_barrier(mem_flags::mem_device)
#define __threadfence_block() threadgroup_barrier(mem_flags::mem_threadgroup)

// CUDA atomic operations
template<typename T>
METAL_FUNC T atomicAdd(device atomic_uint* addr, T val) {
    return atomic_fetch_add_explicit(addr, val, memory_order_relaxed);
}

template<typename T>
METAL_FUNC T atomicMax(device atomic_uint* addr, T val) {
    return atomic_fetch_max_explicit(addr, val, memory_order_relaxed);
}

// CUDA math functions
#define __fdividef(x, y) ((x) / (y))
#define __expf(x) metal::exp(x)
#define __logf(x) metal::log(x)
#define __powf(x, y) metal::pow(x, y)

// SIMD group operations
#define METAL_WARP_SIZE 32
#define warpSize METAL_WARP_SIZE

METAL_FUNC uint get_lane_id() {
    return threadIdx_x & (METAL_WARP_SIZE - 1);
}

METAL_FUNC uint get_warp_id() {
    return threadIdx_x >> 5;
}

// Memory space qualifiers
#define __shared__ threadgroup
#define __constant__ constant
#define __device__ device

#endif /* CUDAMetalKernel_h */