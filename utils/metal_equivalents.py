from typing import Dict, Callable, Any, List, Optional
from .cuda_builtin_functions import CudaBuiltinFunction, CUDA_BUILTIN_FUNCTIONS
from .cuda_to_metal_type_mapping import map_cuda_type_to_metal

class MetalEquivalent:
    def __init__(self, cuda_function: str, metal_function: str,
                 argument_transformer: Optional[Callable[[List[str]], List[str]]] = None,
                 return_transformer: Optional[Callable[[str], str]] = None,
                 requires_custom_implementation: bool = False):
        self.cuda_function = cuda_function
        self.metal_function = metal_function
        self.argument_transformer = argument_transformer
        self.return_transformer = return_transformer
        self.requires_custom_implementation = requires_custom_implementation

    def transform_arguments(self, args: List[str]) -> List[str]:
        if self.argument_transformer:
            return self.argument_transformer(args)
        return args

    def transform_return(self, return_value: str) -> str:
        if self.return_transformer:
            return self.return_transformer(return_value)
        return return_value

def threadIdx_transformer(args: List[str]) -> List[str]:
    return ['thread_position_in_threadgroup']

def blockIdx_transformer(args: List[str]) -> List[str]:
    return ['threadgroup_position_in_grid']

def atomicAdd_transformer(args: List[str]) -> List[str]:
    return [f'atomic_fetch_add_explicit({args[0]}, {args[1]}, memory_order_relaxed)']

METAL_EQUIVALENTS: Dict[str, MetalEquivalent] = {
    'threadIdx': MetalEquivalent('threadIdx', 'thread_position_in_threadgroup', threadIdx_transformer),
    'blockIdx': MetalEquivalent('blockIdx', 'threadgroup_position_in_grid', blockIdx_transformer),
    'blockDim': MetalEquivalent('blockDim', 'threadgroup_size'),
    'gridDim': MetalEquivalent('gridDim', 'grid_size'),
    '__syncthreads': MetalEquivalent('__syncthreads', 'threadgroup_barrier(metal::mem_flags::mem_device)'),
    'atomicAdd': MetalEquivalent('atomicAdd', 'atomic_fetch_add_explicit', atomicAdd_transformer),
    'cudaMalloc': MetalEquivalent('cudaMalloc', 'device.makeBuffer', requires_custom_implementation=True),
    'cudaFree': MetalEquivalent('cudaFree', '', requires_custom_implementation=True),  # No direct equivalent, memory management is different
    'cudaMemcpy': MetalEquivalent('cudaMemcpy', 'memcpy', requires_custom_implementation=True),
}

def get_metal_equivalent(cuda_function: str) -> MetalEquivalent:
    if cuda_function in METAL_EQUIVALENTS:
        return METAL_EQUIVALENTS[cuda_function]

    # For CUDA built-in functions not explicitly defined in METAL_EQUIVALENTS
    if cuda_function in CUDA_BUILTIN_FUNCTIONS:
        cuda_builtin = CUDA_BUILTIN_FUNCTIONS[cuda_function]
        return MetalEquivalent(cuda_function, cuda_builtin.metal_equivalent)

    # If no equivalent is found, return the original function name
    return MetalEquivalent(cuda_function, cuda_function)

def translate_cuda_call_to_metal(cuda_function: str, args: List[str]) -> str:
    equivalent = get_metal_equivalent(cuda_function)
    transformed_args = equivalent.transform_arguments(args)

    if equivalent.requires_custom_implementation:
        return f"// TODO: Implement custom Metal equivalent for {cuda_function}\n" \
               f"// {equivalent.metal_function}({', '.join(transformed_args)})"

    return f"{equivalent.metal_function}({', '.join(transformed_args)})"

def get_metal_type(cuda_type: str) -> str:
    return map_cuda_type_to_metal(cuda_type)

def generate_metal_kernel_signature(kernel_name: str, parameters: List[CudaBuiltinFunction]) -> str:
    metal_params = []
    for i, param in enumerate(parameters):
        metal_type = get_metal_type(param.return_type)
        metal_params.append(f"{metal_type} {param.name} [[buffer({i})]]")

    return f"kernel void {kernel_name}({', '.join(metal_params)})"

