from typing import Dict, Optional

class TypeMapping:
    def __init__(self, cuda_type: str, metal_type: str,
                 requires_header: bool = False,
                 metal_header: Optional[str] = None):
        self.cuda_type = cuda_type
        self.metal_type = metal_type
        self.requires_header = requires_header
        self.metal_header = metal_header

    def __str__(self):
        return f"{self.cuda_type} -> {self.metal_type}"

CUDA_TO_METAL_TYPE_MAP: Dict[str, TypeMapping] = {
    # Integer types
    'char': TypeMapping('char', 'char'),
    'signed char': TypeMapping('signed char', 'char'),
    'unsigned char': TypeMapping('unsigned char', 'uchar'),
    'short': TypeMapping('short', 'short'),
    'unsigned short': TypeMapping('unsigned short', 'ushort'),
    'int': TypeMapping('int', 'int'),
    'unsigned int': TypeMapping('unsigned int', 'uint'),
    'long': TypeMapping('long', 'int'),  # In Metal, long is 32-bit
    'unsigned long': TypeMapping('unsigned long', 'uint'),
    'long long': TypeMapping('long long', 'long'),  # In Metal, long long is 64-bit
    'unsigned long long': TypeMapping('unsigned long long', 'ulong'),

    # Floating-point types
    'float': TypeMapping('float', 'float'),
    'double': TypeMapping('double', 'float'),  # Metal doesn't support double, use float

    # Vector types
    'char2': TypeMapping('char2', 'char2', True, '<metal_simdgroup>'),
    'char3': TypeMapping('char3', 'char3', True, '<metal_simdgroup>'),
    'char4': TypeMapping('char4', 'char4', True, '<metal_simdgroup>'),
    'uchar2': TypeMapping('uchar2', 'uchar2', True, '<metal_simdgroup>'),
    'uchar3': TypeMapping('uchar3', 'uchar3', True, '<metal_simdgroup>'),
    'uchar4': TypeMapping('uchar4', 'uchar4', True, '<metal_simdgroup>'),
    'short2': TypeMapping('short2', 'short2', True, '<metal_simdgroup>'),
    'short3': TypeMapping('short3', 'short3', True, '<metal_simdgroup>'),
    'short4': TypeMapping('short4', 'short4', True, '<metal_simdgroup>'),
    'ushort2': TypeMapping('ushort2', 'ushort2', True, '<metal_simdgroup>'),
    'ushort3': TypeMapping('ushort3', 'ushort3', True, '<metal_simdgroup>'),
    'ushort4': TypeMapping('ushort4', 'ushort4', True, '<metal_simdgroup>'),
    'int2': TypeMapping('int2', 'int2', True, '<metal_simdgroup>'),
    'int3': TypeMapping('int3', 'int3', True, '<metal_simdgroup>'),
    'int4': TypeMapping('int4', 'int4', True, '<metal_simdgroup>'),
    'uint2': TypeMapping('uint2', 'uint2', True, '<metal_simdgroup>'),
    'uint3': TypeMapping('uint3', 'uint3', True, '<metal_simdgroup>'),
    'uint4': TypeMapping('uint4', 'uint4', True, '<metal_simdgroup>'),
    'float2': TypeMapping('float2', 'float2', True, '<metal_simdgroup>'),
    'float3': TypeMapping('float3', 'float3', True, '<metal_simdgroup>'),
    'float4': TypeMapping('float4', 'float4', True, '<metal_simdgroup>'),

    # CUDA-specific types
    'dim3': TypeMapping('dim3', 'uint3', True, '<metal_simdgroup>'),
    'cudaError_t': TypeMapping('cudaError_t', 'int'),
    'cudaStream_t': TypeMapping('cudaStream_t', 'metal::command_queue'),
    'cudaEvent_t': TypeMapping('cudaEvent_t', 'metal::event'),
}

def map_cuda_type_to_metal(cuda_type: str) -> str:
    mapping = CUDA_TO_METAL_TYPE_MAP.get(cuda_type)
    return mapping.metal_type if mapping else cuda_type

def requires_metal_header(cuda_type: str) -> bool:
    mapping = CUDA_TO_METAL_TYPE_MAP.get(cuda_type)
    return mapping.requires_header if mapping else False

def get_metal_header(cuda_type: str) -> Optional[str]:
    mapping = CUDA_TO_METAL_TYPE_MAP.get(cuda_type)
    return mapping.metal_header if mapping else None

def is_vector_type(type_name: str) -> bool:
    return type_name.lower() in [
        'char2', 'char3', 'char4',
        'uchar2', 'uchar3', 'uchar4',
        'short2', 'short3', 'short4',
        'ushort2', 'ushort3', 'ushort4',
        'int2', 'int3', 'int4',
        'uint2', 'uint3', 'uint4',
        'float2', 'float3', 'float4'
    ]

def get_vector_component_type(vector_type: str) -> str:
    base_type = vector_type.rstrip('234')
    return map_cuda_type_to_metal(base_type)

def get_vector_size(vector_type: str) -> int:
    return int(vector_type[-1]) if vector_type[-1].isdigit() else 0