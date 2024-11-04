#some values must be recheked, mackintosh and hackintosh in the futur
# utils/mapping_tables.py

from typing import Dict, Set, Tuple, Optional, Union, List
from enum import Enum
from dataclasses import dataclass
import logging

from .error_handler import CudaTranslationError
from .logger import get_logger

logger = get_logger(__name__)

@dataclass
class MetalType:
    """Metal type information with full metadata"""
    name: str
    size: int
    alignment: int
    can_atomic: bool = False
    texture_format: Optional[str] = None
    sampler_type: Optional[str] = None
    allow_threadgroup: bool = True
    is_builtin: bool = False

@dataclass
class MetalFunction:
    """Metal function metadata"""
    name: str
    return_type: str
    arg_types: List[str]
    has_fast_variant: bool = False
    needs_explicit_cast: bool = False

# Complete Metal type mappings
METAL_TYPES = {
    # Scalar Types
    'bool': MetalType('bool', 1, 1),
    'char': MetalType('char', 1, 1),
    'uchar': MetalType('uchar', 1, 1),
    'short': MetalType('short', 2, 2),
    'ushort': MetalType('ushort', 2, 2),
    'int': MetalType('int', 4, 4, can_atomic=True),
    'uint': MetalType('uint', 4, 4, can_atomic=True),
    'long': MetalType('long', 8, 8),
    'ulong': MetalType('ulong', 8, 8),
    'half': MetalType('half', 2, 2),
    'float': MetalType('float', 4, 4),

    # Vector Types
    'char2': MetalType('char2', 2, 2),
    'char3': MetalType('char3', 4, 4),
    'char4': MetalType('char4', 4, 4),
    'uchar2': MetalType('uchar2', 2, 2),
    'uchar3': MetalType('uchar3', 4, 4),
    'uchar4': MetalType('uchar4', 4, 4),
    'short2': MetalType('short2', 4, 4),
    'short3': MetalType('short3', 8, 8),
    'short4': MetalType('short4', 8, 8),
    'ushort2': MetalType('ushort2', 4, 4),
    'ushort3': MetalType('ushort3', 8, 8),
    'ushort4': MetalType('ushort4', 8, 8),
    'int2': MetalType('int2', 8, 8),
    'int3': MetalType('int3', 16, 16),
    'int4': MetalType('int4', 16, 16),
    'uint2': MetalType('uint2', 8, 8),
    'uint3': MetalType('uint3', 16, 16),
    'uint4': MetalType('uint4', 16, 16),
    'float2': MetalType('float2', 8, 8),
    'float3': MetalType('float3', 16, 16),
    'float4': MetalType('float4', 16, 16),
    'half2': MetalType('half2', 4, 4),
    'half3': MetalType('half3', 8, 8),
    'half4': MetalType('half4', 8, 8),

    # Matrix Types
    'float2x2': MetalType('float2x2', 16, 8),
    'float2x3': MetalType('float2x3', 24, 8),
    'float2x4': MetalType('float2x4', 32, 8),
    'float3x2': MetalType('float3x2', 24, 8),
    'float3x3': MetalType('float3x3', 36, 8),
    'float3x4': MetalType('float3x4', 48, 8),
    'float4x2': MetalType('float4x2', 32, 8),
    'float4x3': MetalType('float4x3', 48, 8),
    'float4x4': MetalType('float4x4', 64, 8),

    # Texture Types
    'texture1d': MetalType('texture1d<float>', 8, 8, texture_format='float'),
    'texture2d': MetalType('texture2d<float>', 8, 8, texture_format='float'),
    'texture3d': MetalType('texture3d<float>', 8, 8, texture_format='float'),
    'texturecube': MetalType('texturecube<float>', 8, 8, texture_format='float'),

    # Sampler Types
    'sampler': MetalType('sampler', 8, 8, sampler_type='sampler'),

    # Atomic Types
    'atomic_int': MetalType('atomic_int', 4, 4, can_atomic=True),
    'atomic_uint': MetalType('atomic_uint', 4, 4, can_atomic=True),

    # SIMD Types
    'simd_float4': MetalType('simd_float4', 16, 16, is_builtin=True),
    'simd_int4': MetalType('simd_int4', 16, 16, is_builtin=True),
    'simd_uint4': MetalType('simd_uint4', 16, 16, is_builtin=True),
}

# Complete Metal function mappings
METAL_FUNCTIONS = {
    # Math Functions
    'sin': MetalFunction('metal::sin', 'float', ['float'], has_fast_variant=True),
    'cos': MetalFunction('metal::cos', 'float', ['float'], has_fast_variant=True),
    'tan': MetalFunction('metal::tan', 'float', ['float'], has_fast_variant=True),
    'asin': MetalFunction('metal::asin', 'float', ['float']),
    'acos': MetalFunction('metal::acos', 'float', ['float']),
    'atan': MetalFunction('metal::atan', 'float', ['float']),
    'sinh': MetalFunction('metal::sinh', 'float', ['float']),
    'cosh': MetalFunction('metal::cosh', 'float', ['float']),
    'tanh': MetalFunction('metal::tanh', 'float', ['float']),
    'exp': MetalFunction('metal::exp', 'float', ['float'], has_fast_variant=True),
    'exp2': MetalFunction('metal::exp2', 'float', ['float'], has_fast_variant=True),
    'log': MetalFunction('metal::log', 'float', ['float'], has_fast_variant=True),
    'log2': MetalFunction('metal::log2', 'float', ['float'], has_fast_variant=True),
    'log10': MetalFunction('metal::log10', 'float', ['float']),
    'pow': MetalFunction('metal::pow', 'float', ['float', 'float'], has_fast_variant=True),
    'sqrt': MetalFunction('metal::sqrt', 'float', ['float'], has_fast_variant=True),
    'rsqrt': MetalFunction('metal::rsqrt', 'float', ['float'], has_fast_variant=True),
    'abs': MetalFunction('metal::abs', 'float', ['float']),
    'min': MetalFunction('metal::min', 'float', ['float', 'float']),
    'max': MetalFunction('metal::max', 'float', ['float', 'float']),
    'ceil': MetalFunction('metal::ceil', 'float', ['float']),
    'floor': MetalFunction('metal::floor', 'float', ['float']),
    'fract': MetalFunction('metal::fract', 'float', ['float']),
    'mod': MetalFunction('metal::fmod', 'float', ['float', 'float']),

    # Atomic Functions
    'atomic_store': MetalFunction('atomic_store_explicit', 'void', ['atomic_type*', 'T'], needs_explicit_cast=True),
    'atomic_load': MetalFunction('atomic_load_explicit', 'T', ['atomic_type*'], needs_explicit_cast=True),
    'atomic_exchange': MetalFunction('atomic_exchange_explicit', 'T', ['atomic_type*', 'T'], needs_explicit_cast=True),
    'atomic_compare_exchange_weak': MetalFunction('atomic_compare_exchange_weak_explicit', 'bool', ['atomic_type*', 'T*', 'T'], needs_explicit_cast=True),
    'atomic_fetch_add': MetalFunction('atomic_fetch_add_explicit', 'T', ['atomic_type*', 'T'], needs_explicit_cast=True),
    'atomic_fetch_sub': MetalFunction('atomic_fetch_sub_explicit', 'T', ['atomic_type*', 'T'], needs_explicit_cast=True),
    'atomic_fetch_and': MetalFunction('atomic_fetch_and_explicit', 'T', ['atomic_type*', 'T'], needs_explicit_cast=True),
    'atomic_fetch_or': MetalFunction('atomic_fetch_or_explicit', 'T', ['atomic_type*', 'T'], needs_explicit_cast=True),
    'atomic_fetch_xor': MetalFunction('atomic_fetch_xor_explicit', 'T', ['atomic_type*', 'T'], needs_explicit_cast=True),

    # Synchronization Functions
    'threadgroup_barrier': MetalFunction('threadgroup_barrier', 'void', ['mem_flags']),
    'simd_barrier': MetalFunction('simd_barrier', 'void', []),

    # SIMD Functions
    'simd_sum': MetalFunction('simd_sum', 'T', ['T']),
    'simd_min': MetalFunction('simd_min', 'T', ['T']),
    'simd_max': MetalFunction('simd_max', 'T', ['T']),
    'simd_and': MetalFunction('simd_and', 'T', ['T']),
    'simd_or': MetalFunction('simd_or', 'T', ['T']),
    'simd_xor': MetalFunction('simd_xor', 'T', ['T']),
    'simd_broadcast': MetalFunction('simd_broadcast', 'T', ['T', 'uint']),
    'simd_shuffle': MetalFunction('simd_shuffle', 'T', ['T', 'uint']),
    'simd_shuffle_xor': MetalFunction('simd_shuffle_xor', 'T', ['T', 'uint']),
    'simd_all': MetalFunction('simd_all', 'bool', ['bool']),
    'simd_any': MetalFunction('simd_any', 'bool', ['bool']),
}

# Complete Metal qualifier mappings
METAL_QUALIFIERS = {
    'kernel': 'kernel',
    'device': 'device',
    'constant': 'constant',
    'threadgroup': 'threadgroup',
    'thread': 'thread',
    'inline': 'inline',
    'static': 'static',
    'volatile': 'volatile',
    'restrict': 'restrict',
    'const': 'const',
    'read_write': 'read_write',
    'read': 'read',
    'write': 'write',
}

# Complete Metal attribute mappings
METAL_ATTRIBUTES = {
    # Buffer binding
    'buffer': '[[buffer(%d)]]',
    'texture': '[[texture(%d)]]',
    'sampler': '[[sampler(%d)]]',

    # Thread position
    'thread_position_in_grid': '[[thread_position_in_grid]]',
    'thread_position_in_threadgroup': '[[thread_position_in_threadgroup]]',
    'threadgroup_position_in_grid': '[[threadgroup_position_in_grid]]',
    'threads_per_threadgroup': '[[threads_per_threadgroup]]',
    'threadgroups_per_grid': '[[threadgroups_per_grid]]',
    'thread_index_in_simdgroup': '[[thread_index_in_simdgroup]]',
    'simdgroup_index_in_threadgroup': '[[simdgroup_index_in_threadgroup]]',

    # Function attributes
    'always_inline': '[[always_inline]]',
    'noinline': '[[noinline]]',
    'convergent': '[[convergent]]',

    # Memory attributes
    'packed': '[[packed]]',
    'aligned': '[[aligned(%d)]]',
}

# Memory flag mappings
METAL_MEMORY_FLAGS = {
    'mem_none': 'mem_flags::mem_none',
    'mem_device': 'mem_flags::mem_device',
    'mem_threadgroup': 'mem_flags::mem_threadgroup',
    'mem_texture': 'mem_flags::mem_texture',
}

# Complete Metal texture formats
METAL_TEXTURE_FORMATS = {
    'R8Unorm': {'size': 1, 'components': 1, 'type': 'unorm8'},
    'RG8Unorm': {'size': 2, 'components': 2, 'type': 'unorm8'},
    'RGBA8Unorm': {'size': 4, 'components': 4, 'type': 'unorm8'},
    'R16Float': {'size': 2, 'components': 1, 'type': 'float16'},
    'RG16Float': {'size': 4, 'components': 2, 'type': 'float16'},
    'RGBA16Float': {'size': 8, 'components': 4, 'type': 'float16'},
    'R32Float': {'size': 4, 'components': 1, 'type': 'float32'},
    'RG32Float': {'size': 8, 'components': 2, 'type': 'float32'},
    'RGBA32Float': {'size': 16, 'components': 4, 'type': 'float32'},
    'R8Sint': {'size': 1, 'components': 1, 'type': 'sint8'},
    'RG8Sint': {'size': 2, 'components': 2, 'type': 'sint8'},
    'RGBA8Sint': {'size': 4, 'components': 4, 'type': 'sint8'},
    'R16Sint': {'size': 2, 'components': 1, 'type': 'sint16'},
    'RG16Sint': {'size': 4, 'components': 2, 'type': 'sint16'},
    'RGBA16Sint': {'size': 8, 'components': 4, 'type': 'sint16'},
    'R32Sint': {'size': 4, 'components': 1, 'type': 'sint32'},
    'RG32Sint': {'size': 8, 'components': 2, 'type': 'sint32'},
    'RGBA32Sint': {'size': 16, 'components': 4, 'type': 'sint32'},
}

# Address space mappings
METAL_ADDRESS_SPACES = {
    'default': '',
    'device': 'device',
    'constant': 'constant',
    'threadgroup': 'threadgroup',
    'thread': 'thread',
}
# Address space semantics
METAL_ADDRESS_SPACE_SEMANTICS = {
    'device': {
        'access': 'read_write',
        'scope': 'device',
        'alignment': 16,
        'cache_mode': 'cached',
        'can_alias': True
    },
    'constant': {
        'access': 'read',
        'scope': 'device',
        'alignment': 16,
        'cache_mode': 'cached',
        'can_alias': False
    },
    'threadgroup': {
        'access': 'read_write',
        'scope': 'threadgroup',
        'alignment': 16,
        'cache_mode': 'cached',
        'can_alias': True
    },
    'thread': {
        'access': 'read_write',
        'scope': 'thread',
        'alignment': 16,
        'cache_mode': 'none',
        'can_alias': True
    }
}

# Memory order mappings
METAL_MEMORY_ORDERS = {
    'relaxed': 'memory_order_relaxed',
    'acquire': 'memory_order_acquire',
    'release': 'memory_order_release',
    'acq_rel': 'memory_order_acq_rel',
    'seq_cst': 'memory_order_seq_cst'
}

# Memory scope mappings
METAL_MEMORY_SCOPES = {
    'device': 'memory_scope_device',
    'threadgroup': 'memory_scope_threadgroup',
    'simdgroup': 'memory_scope_simdgroup'
}

# Attribute argument mappings
METAL_ATTRIBUTE_ARGUMENTS = {
    'buffer': lambda idx: f'[[buffer({idx})]]',
    'texture': lambda idx: f'[[texture({idx})]]',
    'sampler': lambda idx: f'[[sampler({idx})]]',
    'thread_position_in_grid': lambda: '[[thread_position_in_grid]]',
    'threadgroup_position_in_grid': lambda: '[[threadgroup_position_in_grid]]',
    'threads_per_threadgroup': lambda: '[[threads_per_threadgroup]]',
    'thread_position_in_threadgroup': lambda: '[[thread_position_in_threadgroup]]',
    'thread_index_in_simdgroup': lambda: '[[thread_index_in_simdgroup]]',
    'simdgroup_index_in_threadgroup': lambda: '[[simdgroup_index_in_threadgroup]]'
}

# Resource binding mappings
METAL_RESOURCE_BINDINGS = {
    'buffer': {
        'max_per_stage': 31,
        'alignment': 256,
        'offset_alignment': 256,
        'min_size': 16,
    },
    'texture': {
        'max_per_stage': 128,
        'max_arrays': 32,
        'alignment': 16,
    },
    'sampler': {
        'max_per_stage': 16,
        'alignment': 8,
    }
}

# Texture access mappings
METAL_TEXTURE_ACCESS = {
    'sample': 'access::sample',
    'read': 'access::read',
    'write': 'access::write',
    'read_write': 'access::read_write'
}

# Sampler state mappings
METAL_SAMPLER_STATES = {
    'address_modes': {
        'clamp_to_edge': 'address::clamp_to_edge',
        'repeat': 'address::repeat',
        'mirrored_repeat': 'address::mirrored_repeat',
        'clamp_to_zero': 'address::clamp_to_zero',
        'clamp_to_border': 'address::clamp_to_border'
    },
    'min_filter': {
        'nearest': 'filter::nearest',
        'linear': 'filter::linear'
    },
    'mag_filter': {
        'nearest': 'filter::nearest',
        'linear': 'filter::linear'
    },
    'mip_filter': {
        'none': 'filter::none',
        'nearest': 'filter::nearest',
        'linear': 'filter::linear'
    },
    'compare_func': {
        'never': 'compare_func::never',
        'less': 'compare_func::less',
        'less_equal': 'compare_func::less_equal',
        'greater': 'compare_func::greater',
        'greater_equal': 'compare_func::greater_equal',
        'equal': 'compare_func::equal',
        'not_equal': 'compare_func::not_equal',
        'always': 'compare_func::always'
    }
}

# Thread mapping details
METAL_THREAD_MAPPING = {
    'simd_width': 32,
    'max_threads_per_threadgroup': 1024,
    'max_threadgroups_per_grid': (2**16 - 1, 2**16 - 1, 2**16 - 1),
    'max_total_threadgroup_memory': 32768,  # 32KB
    'preferred_threadgroup_size_multiple': 32
}

# Builtin function variants
METAL_BUILTIN_VARIANTS = {
    'precise': {
        'prefix': 'metal::',
        'performance': 'high_precision',
        'available': True
    },
    'fast': {
        'prefix': 'metal::fast::',
        'performance': 'high_performance',
        'available': True
    },
    'native': {
        'prefix': 'metal::native::',
        'performance': 'maximum_performance',
        'available': True
    }
}

class MetalMappingRegistry:
    """Registry for Metal mappings with validation and optimization."""

    def __init__(self):
        self._types = METAL_TYPES
        self._functions = METAL_FUNCTIONS
        self._qualifiers = METAL_QUALIFIERS
        self._attributes = METAL_ATTRIBUTES
        self._memory_flags = METAL_MEMORY_FLAGS
        self._texture_formats = METAL_TEXTURE_FORMATS
        self._address_spaces = METAL_ADDRESS_SPACES
        self._sampler_states = METAL_SAMPLER_STATES
        self._thread_mapping = METAL_THREAD_MAPPING
        self._builtin_variants = METAL_BUILTIN_VARIANTS

    def get_metal_type(self, cuda_type: str) -> Optional[MetalType]:
        """Get Metal type equivalent for CUDA type."""
        return self._types.get(cuda_type.lower())

    def get_metal_function(self, cuda_function: str) -> Optional[MetalFunction]:
        """Get Metal function equivalent for CUDA function."""
        return self._functions.get(cuda_function)

    def get_metal_qualifier(self, cuda_qualifier: str) -> Optional[str]:
        """Get Metal qualifier equivalent for CUDA qualifier."""
        return self._qualifiers.get(cuda_qualifier.lower())

    def get_metal_attribute(self, cuda_attribute: str, *args) -> Optional[str]:
        """Get Metal attribute with arguments."""
        attr_template = self._attributes.get(cuda_attribute)
        if not attr_template:
            return None
        try:
            return attr_template % args if args else attr_template
        except TypeError:
            logger.error(f"Invalid arguments for attribute {cuda_attribute}: {args}")
            return None

    def get_texture_format(self, format_name: str) -> Optional[Dict]:
        """Get Metal texture format details."""
        return self._texture_formats.get(format_name)

    def get_address_space(self, cuda_space: str) -> Optional[str]:
        """Get Metal address space equivalent."""
        return self._address_spaces.get(cuda_space.lower())

    def get_sampler_state(self, parameter: str, value: str) -> Optional[str]:
        """Get Metal sampler state equivalent."""
        param_dict = self._sampler_states.get(parameter)
        if param_dict:
            return param_dict.get(value.lower())
        return None

    def get_thread_limit(self, dimension: str) -> Optional[int]:
        """Get Metal thread limits."""
        return self._thread_mapping.get(dimension)

    def get_function_variant(self, function_name: str, variant: str = 'precise') -> Optional[str]:
        """Get Metal function variant."""
        variant_info = self._builtin_variants.get(variant)
        if not variant_info or not variant_info['available']:
            return None
        return f"{variant_info['prefix']}{function_name}"

    def validate_metal_compatibility(self, cuda_type: str) -> bool:
        """Validate if CUDA type has Metal equivalent."""
        return cuda_type.lower() in self._types

    def get_optimal_alignment(self, metal_type: MetalType) -> int:
        """Get optimal alignment for Metal type."""
        if metal_type.texture_format:
            return METAL_RESOURCE_BINDINGS['texture']['alignment']
        if metal_type.sampler_type:
            return METAL_RESOURCE_BINDINGS['sampler']['alignment']
        return max(metal_type.alignment, METAL_RESOURCE_BINDINGS['buffer']['alignment'])

    def get_memory_order(self, cuda_order: str) -> str:
        """Get Metal memory order equivalent."""
        return METAL_MEMORY_ORDERS.get(cuda_order.lower(), 'memory_order_relaxed')

    def get_memory_scope(self, cuda_scope: str) -> str:
        """Get Metal memory scope equivalent."""
        return METAL_MEMORY_SCOPES.get(cuda_scope.lower(), 'memory_scope_device')

logger.info("MetalMappingRegistry initialized with complete mappings")
