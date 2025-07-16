from typing import Dict, List, Tuple

class CudaBuiltinFunction:
    def __init__(self, name: str, return_type: str, parameters: List[Tuple[str, str]],
                 is_device_function: bool, metal_equivalent: str):
        self.name = name
        self.return_type = return_type
        self.parameters = parameters
        self.is_device_function = is_device_function
        self.metal_equivalent = metal_equivalent

    def __str__(self):
        params_str = ', '.join([f'{param_type} {param_name}' for param_name, param_type in self.parameters])
        return f'{self.return_type} {self.name}({params_str})'

CUDA_BUILTIN_FUNCTIONS: Dict[str, CudaBuiltinFunction] = {
    # Thread Management
    'threadIdx': CudaBuiltinFunction('threadIdx', 'uint3', [], True, 'thread_position_in_threadgroup'),
    'blockIdx': CudaBuiltinFunction('blockIdx', 'uint3', [], True, 'threadgroup_position_in_grid'),
    'blockDim': CudaBuiltinFunction('blockDim', 'uint3', [], True, 'threadgroup_size'),
    'gridDim': CudaBuiltinFunction('gridDim', 'uint3', [], True, 'grid_size'),
    'warpSize': CudaBuiltinFunction('warpSize', 'int', [], True, '32'),

    # Synchronization
    '__syncthreads': CudaBuiltinFunction('__syncthreads', 'void', [], True, 'threadgroup_barrier(mem_flags::mem_device)'),
    '__syncwarp': CudaBuiltinFunction('__syncwarp', 'void', [('mask', 'unsigned int')], True, 'simdgroup_barrier(mem_flags::mem_none)'),

    # Atomic Operations
    'atomicAdd': CudaBuiltinFunction('atomicAdd', 'T', [('address', 'T*'), ('val', 'T')], True, 'atomic_fetch_add_explicit'),
    'atomicSub': CudaBuiltinFunction('atomicSub', 'T', [('address', 'T*'), ('val', 'T')], True, 'atomic_fetch_sub_explicit'),
    'atomicExch': CudaBuiltinFunction('atomicExch', 'T', [('address', 'T*'), ('val', 'T')], True, 'atomic_exchange_explicit'),
    'atomicMin': CudaBuiltinFunction('atomicMin', 'T', [('address', 'T*'), ('val', 'T')], True, 'atomic_fetch_min_explicit'),
    'atomicMax': CudaBuiltinFunction('atomicMax', 'T', [('address', 'T*'), ('val', 'T')], True, 'atomic_fetch_max_explicit'),
    'atomicInc': CudaBuiltinFunction('atomicInc', 'unsigned int', [('address', 'unsigned int*'), ('val', 'unsigned int')], True, 'custom_atomic_inc'),
    'atomicDec': CudaBuiltinFunction('atomicDec', 'unsigned int', [('address', 'unsigned int*'), ('val', 'unsigned int')], True, 'custom_atomic_dec'),
    'atomicCAS': CudaBuiltinFunction('atomicCAS', 'T', [('address', 'T*'), ('compare', 'T'), ('val', 'T')], True, 'atomic_compare_exchange_weak_explicit'),

    # Math Functions (subset)
    'sin': CudaBuiltinFunction('sin', 'float', [('x', 'float')], False, 'sin'),
    'cos': CudaBuiltinFunction('cos', 'float', [('x', 'float')], False, 'cos'),
    'exp': CudaBuiltinFunction('exp', 'float', [('x', 'float')], False, 'exp'),
    'log': CudaBuiltinFunction('log', 'float', [('x', 'float')], False, 'log'),
    'sqrt': CudaBuiltinFunction('sqrt', 'float', [('x', 'float')], False, 'sqrt'),

    # Vector Types
    'make_int2': CudaBuiltinFunction('make_int2', 'int2', [('x', 'int'), ('y', 'int')], False, 'int2'),
    'make_float2': CudaBuiltinFunction('make_float2', 'float2', [('x', 'float'), ('y', 'float')], False, 'float2'),

    # Texture Functions
    'tex2D': CudaBuiltinFunction('tex2D', 'float4', [('texObj', 'texture<T, 2>'), ('x', 'float'), ('y', 'float')], True, 'sample'),

    # Memory Management
    'cudaMalloc': CudaBuiltinFunction('cudaMalloc', 'cudaError_t', [('devPtr', 'void**'), ('size', 'size_t')], False, 'device.makeBuffer'),
    'cudaFree': CudaBuiltinFunction('cudaFree', 'cudaError_t', [('devPtr', 'void*')], False, 'None'),
    'cudaMemcpy': CudaBuiltinFunction('cudaMemcpy', 'cudaError_t', [('dst', 'void*'), ('src', 'const void*'), ('count', 'size_t'), ('kind', 'cudaMemcpyKind')], False, 'memcpy'),
}

def is_cuda_builtin(func_name: str) -> bool:
    return func_name in CUDA_BUILTIN_FUNCTIONS

def get_cuda_builtin(func_name: str) -> CudaBuiltinFunction:
    return CUDA_BUILTIN_FUNCTIONS.get(func_name)

def get_metal_equivalent(func_name: str) -> str:
    builtin = get_cuda_builtin(func_name)
    return builtin.metal_equivalent if builtin else None

def is_device_function(func_name: str) -> bool:
    builtin = get_cuda_builtin(func_name)
    return builtin.is_device_function if builtin else False

def get_return_type(func_name: str) -> str:
    builtin = get_cuda_builtin(func_name)
    return builtin.return_type if builtin else None

def get_parameters(func_name: str) -> List[Tuple[str, str]]:
    builtin = get_cuda_builtin(func_name)
    return builtin.parameters if builtin else []

