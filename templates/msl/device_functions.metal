#include <metal_stdlib>
using namespace metal;

// Helper function that can be used by kernels
float compute_something(float value) {
    return value * 2.0;
}
