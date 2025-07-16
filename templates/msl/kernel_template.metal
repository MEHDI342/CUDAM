#include <metal_stdlib>
#include "device_functions.metal"
using namespace metal;

kernel void example_kernel(const device float* input [[buffer(0)]],
                           device float* output [[buffer(1)]],
                           uint id [[thread_position_in_grid]]) {
    output[id] = compute_something(input[id]);
}
