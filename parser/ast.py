from typing import List, Dict, Any, Optional, Union, Set, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)

CUDA_TO_METAL_TYPE_MAP = {
    # Basic types
    'float': 'float',
    'double': 'float',  # Metal doesn't support double
    'int': 'int32_t',
    'unsigned int': 'uint32_t',
    'long long': 'int64_t',
    'unsigned long long': 'uint64_t',
    'char': 'int8_t',
    'unsigned char': 'uint8_t',
    'short': 'int16_t',
    'unsigned short': 'uint16_t',
    'bool': 'bool',
    'void': 'void',

    # Vector types
    'float2': 'float2',
    'float3': 'float3',
    'float4': 'float4',
    'int2': 'int2',
    'int3': 'int3',
    'int4': 'int4',
    'uint2': 'uint2',
    'uint3': 'uint3',
    'uint4': 'uint4',

    # CUDA-specific types
    'dim3': 'uint3',
    'size_t': 'size_t',
    'cudaError_t': 'int32_t',
}

CUDA_TO_METAL_OPERATORS = {
    # Arithmetic
    '+': '+',
    '-': '-',
    '*': '*',
    '/': '/',
    '%': '%',

    # Bitwise
    '&': '&',
    '|': '|',
    '^': '^',
    '<<': '<<',
    '>>': '>>',
    '~': '~',

    # Logical
    '&&': '&&',
    '||': '||',
    '!': '!',

    # Comparison
    '==': '==',
    '!=': '!=',
    '<': '<',
    '>': '>',
    '<=': '<=',
    '>=': '>=',

    # Assignment
    '=': '=',
    '+=': '+=',
    '-=': '-=',
    '*=': '*=',
    '/=': '/=',
    '%=': '%=',
    '&=': '&=',
    '|=': '|=',
    '^=': '^=',
    '<<=': '<<=',
    '>>=': '>>=',
}

CUDA_TO_METAL_FUNCTION_MAP = {
    # Math functions
    'sin': 'metal::sin',
    'cos': 'metal::cos',
    'tan': 'metal::tan',
    'asin': 'metal::asin',
    'acos': 'metal::acos',
    'atan': 'metal::atan',
    'sinh': 'metal::sinh',
    'cosh': 'metal::cosh',
    'tanh': 'metal::tanh',
    'exp': 'metal::exp',
    'exp2': 'metal::exp2',
    'log': 'metal::log',
    'log2': 'metal::log2',
    'log10': 'metal::log10',
    'pow': 'metal::pow',
    'sqrt': 'metal::sqrt',
    'rsqrt': 'metal::rsqrt',
    'fabs': 'metal::abs',
    'floor': 'metal::floor',
    'ceil': 'metal::ceil',
    'fmin': 'metal::min',
    'fmax': 'metal::max',

    # Synchronization
    '__syncthreads': 'threadgroup_barrier(mem_flags::mem_device)',
    '__threadfence': 'threadgroup_barrier(mem_flags::mem_device)',
    '__threadfence_block': 'threadgroup_barrier(mem_flags::mem_threadgroup)',

    # Atomic operations
    'atomicAdd': 'atomic_fetch_add_explicit',
    'atomicSub': 'atomic_fetch_sub_explicit',
    'atomicExch': 'atomic_exchange_explicit',
    'atomicMin': 'atomic_fetch_min_explicit',
    'atomicMax': 'atomic_fetch_max_explicit',
    'atomicAnd': 'atomic_fetch_and_explicit',
    'atomicOr': 'atomic_fetch_or_explicit',
    'atomicXor': 'atomic_fetch_xor_explicit',
    'atomicCAS': 'atomic_compare_exchange_weak_explicit',
}

from typing import List, Dict, Any, Optional, Set, Tuple
import logging

logger = logging.getLogger(__name__)

class CudaASTNode:
    """Base class for all AST nodes with enhanced Metal support"""
    def __init__(self, kind: str, spelling: Optional[str] = None, type: Optional[str] = None):
        self.kind = kind
        self.spelling = spelling
        self.type = type
        self.children: List['CudaASTNode'] = []
        self.parent: Optional['CudaASTNode'] = None
        self.source_location: Optional[Dict[str, int]] = None

        # Metal-specific attributes
        self.metal_translation: Optional[str] = None
        self.metal_type: Optional[str] = None
        self.metal_limitations: Dict[str, Any] = {}
        self.optimization_metadata: Dict[str, Any] = {
            'vectorizable': False,
            'coalesced_access': False,
            'requires_simd_group': False,
            'threadgroup_memory_size': 0,
            'atomic_operations': [],
            'barrier_points': []
        }

    def add_child(self, child: 'CudaASTNode') -> None:
        """Add a child node to this node"""
        self.children.append(child)
        child.parent = self

    def set_source_location(self, file: str, line: int, column: int) -> None:
        """Set source code location information"""
        self.source_location = {'file': file, 'line': line, 'column': column}

    def get_metal_translation(self) -> str:
        """Get Metal translation for this node"""
        if self.metal_translation is None:
            self.metal_translation = self._generate_metal_translation()
        return self.metal_translation

    def _generate_metal_translation(self) -> str:
        """Generate Metal translation for this node"""
        raise NotImplementedError("Subclasses must implement _generate_metal_translation")

    def validate_metal_compatibility(self) -> List[str]:
        """Validate node's compatibility with Metal"""
        errors = []
        self._check_metal_limitations(errors)
        return errors

    def _check_metal_limitations(self, errors: List[str]) -> None:
        """Check for Metal-specific limitations"""
        # Base implementation - subclasses should override if needed
        pass

    def optimize_for_metal(self) -> None:
        """Apply Metal-specific optimizations"""
        # Base implementation - apply optimizations recursively
        for child in self.children:
            child.optimize_for_metal()

    def _optimize_memory_access(self) -> None:
        """Optimize memory access patterns"""
        if hasattr(self, 'is_buffer') and getattr(self, 'is_buffer'):
            self.optimization_metadata['coalesced_access'] = True
            self.optimization_metadata['vectorizable'] = self._can_vectorize()

    def _can_vectorize(self) -> bool:
        """Check if the node can be vectorized"""
        vector_types = {'float', 'int', 'uint'}
        return (hasattr(self, 'data_type') and
                getattr(self, 'data_type', '').rstrip('234') in vector_types)

    def get_ancestor_of_type(self, node_type: type) -> Optional['CudaASTNode']:
        """Get the nearest ancestor of a specific type"""
        current = self.parent
        while current is not None:
            if isinstance(current, node_type):
                return current
            current = current.parent
        return None

    def find_children_of_type(self, node_type: type) -> List['CudaASTNode']:
        """Find all children of a specific type"""
        result = []
        for child in self.children:
            if isinstance(child, node_type):
                result.append(child)
            result.extend(child.find_children_of_type(node_type))
        return result

    def get_scope(self) -> str:
        """Get the scope of this node"""
        if hasattr(self, 'metal_scope'):
            return getattr(self, 'metal_scope')
        return self.parent.get_scope() if self.parent else 'global'

    def requires_barrier(self) -> bool:
        """Check if this node requires a barrier"""
        return bool(self.optimization_metadata['barrier_points'])

    def has_side_effects(self) -> bool:
        """Check if this node has side effects"""
        # Base implementation - subclasses should override if needed
        return False

    def get_dependency_info(self) -> Dict[str, Any]:
        """Get dependency information for this node"""
        return {
            'reads': set(),
            'writes': set(),
            'dependencies': [],
            'scope': self.get_scope()
        }

    def __repr__(self) -> str:
        """String representation of the node"""
        return f"{self.__class__.__name__}(kind='{self.kind}', spelling='{self.spelling}')"

class MetalFeatureSet:
    """Metal feature sets and limitations"""
    MAX_THREADS_PER_THREADGROUP = 1024
    MAX_THREADGROUPS_PER_GRID = (2048, 2048, 2048)
    MAX_TOTAL_THREADGROUP_MEMORY = 32768  # 32KB
    MAX_BUFFER_SIZE = 1 << 30  # 1GB
    SIMD_GROUP_SIZE = 32
    PREFERRED_THREADGROUP_SIZE = 256

class CudaBuiltinVariableNode(CudaASTNode):
    """Represents CUDA built-in variables"""
    def __init__(self, name: str):
        super().__init__(kind='CudaBuiltinVariable', spelling=name)
        self.metal_equivalent = self._get_metal_equivalent()

    def _get_metal_equivalent(self) -> str:
        """Get Metal equivalent for CUDA built-in variable"""
        equivalents = {
            'threadIdx': 'thread_position_in_threadgroup',
            'blockIdx': 'threadgroup_position_in_grid',
            'blockDim': 'threads_per_threadgroup',
            'gridDim': 'threadgroups_per_grid',
            'warpSize': str(MetalFeatureSet.SIMD_GROUP_SIZE)
        }
        return equivalents.get(self.spelling, self.spelling)

    def _generate_metal_translation(self) -> str:
        """Generate Metal translation for this built-in variable"""
        return self.metal_equivalent

    def has_side_effects(self) -> bool:
        """Built-in variables have no side effects"""
        return False

    def get_dependency_info(self) -> Dict[str, Any]:
        """Get dependency information for built-in variables"""
        return {
            'reads': {self.spelling},
            'writes': set(),
            'dependencies': [],
            'scope': 'builtin'
        }

class CudaBuiltinVariableNode(CudaASTNode):
    """Represents CUDA built-in variables"""
    def __init__(self, name: str):
        super().__init__(kind='CudaBuiltinVariable', spelling=name)
        self.metal_equivalent = self._get_metal_equivalent()

    def _get_metal_equivalent(self) -> str:
        """Get Metal equivalent for CUDA built-in variable"""
        equivalents = {
            'threadIdx': 'thread_position_in_threadgroup',
            'blockIdx': 'threadgroup_position_in_grid',
            'blockDim': 'threads_per_threadgroup',
            'gridDim': 'threadgroups_per_grid',
            'warpSize': str(MetalFeatureSet.SIMD_GROUP_SIZE)
        }
        return equivalents.get(self.spelling, self.spelling)

    def _generate_metal_translation(self) -> str:
        return self.metal_equivalent

class MetalVersion(Enum):
    """Supported Metal versions for different MacOS releases"""
    MACOS_11 = "Metal 2.3"  # Big Sur
    MACOS_12 = "Metal 2.4"  # Monterey
    MACOS_13 = "Metal 3.0"  # Ventura

class MetalGPUFamily(Enum):
    """Metal GPU families for Apple Silicon"""
    APPLE7 = "apple7"  # M1
    APPLE8 = "apple8"  # M2
    APPLE9 = "apple9"  # M3

class MetalFeatureSet:
    """Metal feature sets and limitations"""
    MAX_THREADS_PER_THREADGROUP = 1024
    MAX_THREADGROUPS_PER_GRID = (2048, 2048, 2048)
    MAX_TOTAL_THREADGROUP_MEMORY = 32768  # 32KB
    MAX_BUFFER_SIZE = 1 << 30  # 1GB
    SIMD_GROUP_SIZE = 32
    PREFERRED_THREADGROUP_SIZE = 256

class CudaASTNode:
    """Base class for all AST nodes with enhanced Metal support"""
    def __init__(self, kind: str, spelling: Optional[str] = None, type: Optional[str] = None):
        self.kind = kind
        self.spelling = spelling
        self.type = type
        self.children: List['CudaASTNode'] = []
        self.parent: Optional['CudaASTNode'] = None
        self.source_location: Optional[Dict[str, int]] = None

        # Metal-specific attributes
        self.metal_translation: Optional[str] = None
        self.metal_type: Optional[str] = None
        self.metal_limitations: Dict[str, Any] = {}
        self.optimization_metadata: Dict[str, Any] = {
            'vectorizable': False,
            'coalesced_access': False,
            'requires_simd_group': False,
            'threadgroup_memory_size': 0,
            'atomic_operations': [],
            'barrier_points': []
        }

    def add_child(self, child: 'CudaASTNode') -> None:
        self.children.append(child)
        child.parent = self

    def set_source_location(self, file: str, line: int, column: int) -> None:
        self.source_location = {'file': file, 'line': line, 'column': column}

    def get_metal_translation(self) -> str:
        """Get Metal translation for this node"""
        if self.metal_translation is None:
            self.metal_translation = self._generate_metal_translation()
        return self.metal_translation

    def _generate_metal_translation(self) -> str:
        """Generate Metal translation for this node"""
        raise NotImplementedError("Subclasses must implement _generate_metal_translation")

    def validate_metal_compatibility(self) -> List[str]:
        """Validate node's compatibility with Metal"""
        errors = []
        self._check_metal_limitations(errors)
        return errors

    def _check_metal_limitations(self, errors: List[str]) -> None:
        """Check for Metal-specific limitations"""
        if isinstance(self, KernelNode):
            if self.thread_count > MetalFeatureSet.MAX_THREADS_PER_THREADGROUP:
                errors.append(f"Thread count {self.thread_count} exceeds Metal maximum of {MetalFeatureSet.MAX_THREADS_PER_THREADGROUP}")
            if self.shared_memory_size > MetalFeatureSet.MAX_TOTAL_THREADGROUP_MEMORY:
                errors.append(f"Shared memory size {self.shared_memory_size} exceeds Metal maximum of {MetalFeatureSet.MAX_TOTAL_THREADGROUP_MEMORY}")

    def optimize_for_metal(self) -> None:
        """Apply Metal-specific optimizations"""
        if isinstance(self, KernelNode):
            self._optimize_kernel()
        elif isinstance(self, VariableNode):
            self._optimize_memory_access()

        for child in self.children:
            child.optimize_for_metal()

    def _optimize_memory_access(self) -> None:
        """Optimize memory access patterns"""
        if hasattr(self, 'is_buffer') and self.is_buffer:
            self.optimization_metadata['coalesced_access'] = True
            self.optimization_metadata['vectorizable'] = self._can_vectorize()

    def _can_vectorize(self) -> bool:
        """Check if the node can be vectorized"""
        vector_types = {'float', 'int', 'uint'}
        return hasattr(self, 'data_type') and self.data_type.rstrip('234') in vector_types

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(kind='{self.kind}', spelling='{self.spelling}')"

class FunctionNode(CudaASTNode):
    """Function node with Metal translation support"""
    def __init__(self, name: str, return_type: str, parameters: List['VariableNode'],
                 body: List[CudaASTNode], function_type: str, attributes: List[str]):
        super().__init__(kind='Function', spelling=name)
        self.return_type = return_type
        self.parameters = parameters
        self.body = body
        self.function_type = function_type
        self.attributes = attributes

        # Metal-specific
        self.metal_return_type = self._map_return_type()
        self.is_device_function = function_type == 'device'

        for param in parameters:
            self.add_child(param)
        for stmt in body:
            self.add_child(stmt)

    def _map_return_type(self) -> str:
        """Map CUDA return type to Metal"""
        return CUDA_TO_METAL_TYPE_MAP.get(self.return_type, self.return_type)

    def _generate_metal_translation(self) -> str:
        params = [param.get_metal_translation() for param in self.parameters]
        param_str = ", ".join(params)
        body_lines = []
        for stmt in self.body:
            trans = stmt.get_metal_translation()
            if trans:
                body_lines.extend(trans.split('\n'))

        body_str = "\n    ".join(body_lines)
        metal_attributes = self._get_metal_attributes()

        return f"{metal_attributes}\n{self.metal_return_type} {self.spelling}({param_str})\n{{\n    {body_str}\n}}"

    def _get_metal_attributes(self) -> str:
        """Get Metal-specific attributes"""
        attrs = []
        if 'device' in self.attributes:
            attrs.append('__attribute__((device))')
        return " ".join(attrs)

class KernelNode(FunctionNode):
    """CUDA kernel function with M1/M2-optimized Metal translation"""
    def __init__(self, name: str, parameters: List['VariableNode'],
                 body: List[CudaASTNode], attributes: List[str],
                 launch_config: Optional[Dict[str, Any]] = None):
        super().__init__(name, 'void', parameters, body, 'kernel', attributes)
        self.launch_config = launch_config or {}

        # Metal-specific attributes
        self.thread_count = 0
        self.shared_memory_size = 0
        self.metal_kernel_name = f"metal_{name}"
        self.threadgroup_size = self._calculate_optimal_threadgroup_size()
        self.requires_simd_group = False
        self.metal_buffer_indices: Dict[str, int] = {}

    def _calculate_optimal_threadgroup_size(self) -> Dict[str, int]:
        """Calculate optimal threadgroup size for M1/M2"""
        if not self.launch_config:
            return {'x': 256, 'y': 1, 'z': 1}

        block_dim = self.launch_config.get('block_dim', {})
        x = min(block_dim.get('x', 256), MetalFeatureSet.MAX_THREADS_PER_THREADGROUP)
        y = min(block_dim.get('y', 1), MetalFeatureSet.MAX_THREADS_PER_THREADGROUP // x)
        z = min(block_dim.get('z', 1), MetalFeatureSet.MAX_THREADS_PER_THREADGROUP // (x * y))

        # Ensure multiple of SIMD group size
        x = ((x + MetalFeatureSet.SIMD_GROUP_SIZE - 1) //
             MetalFeatureSet.SIMD_GROUP_SIZE * MetalFeatureSet.SIMD_GROUP_SIZE)

        return {'x': x, 'y': y, 'z': z}

    def _generate_metal_translation(self) -> str:
        """Generate Metal kernel code"""
        params = self._translate_parameters()
        signature = f"kernel void {self.metal_kernel_name}({', '.join(params)})"

        # Thread indexing
        body = [
            "const uint3 thread_position_in_grid [[thread_position_in_grid]];",
            "const uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]];",
            "const uint3 threadgroup_position [[threadgroup_position_in_grid]];",
            "const uint3 threadgroups_per_grid [[threadgroups_per_grid]];",
            "const uint3 threads_per_threadgroup [[threads_per_threadgroup]];"
        ]

        # SIMD group support
        if self.requires_simd_group:
            body.extend([
                "const uint simd_lane_id = thread_position_in_threadgroup.x & 0x1F;",
                "const uint simd_group_id = thread_position_in_threadgroup.x >> 5;"
            ])

        # Translate body
        for stmt in self.body:
            trans = stmt.get_metal_translation()
            if trans:
                body.extend([line for line in trans.split('\n')])

        body_str = "\n    ".join(body)
        attributes = self._get_metal_attributes()

        return f"{attributes}\n{signature}\n{{\n    {body_str}\n}}"

    def _translate_parameters(self) -> List[str]:
        """Translate kernel parameters to Metal"""
        metal_params = []
        for idx, param in enumerate(self.parameters):
            metal_type = param.get_metal_type()

            if param.is_buffer():
                qualifier = "device" if not param.is_readonly() else "constant"
                metal_params.append(f"{qualifier} {metal_type}* {param.spelling} [[buffer({idx})]]")
            elif param.is_texture():
                metal_params.append(f"texture2d<float, access::read> {param.spelling} [[texture({idx})]]")
            else:
                metal_params.append(f"constant {metal_type}& {param.spelling} [[buffer({idx})]]")

            self.metal_buffer_indices[param.spelling] = idx

        return metal_params

    def _get_metal_attributes(self) -> str:
        """Generate Metal kernel attributes"""
        attrs = []

        tg_size = self.threadgroup_size
        attrs.append(f"[[threads_per_threadgroup({tg_size['x']}, {tg_size['y']}, {tg_size['z']})]]")

        if self.thread_count > 0:
            attrs.append(f"[[max_total_threads_per_threadgroup({self.thread_count})]]")

        return " ".join(attrs)

    def _optimize_kernel(self) -> None:
        """Apply kernel-specific optimizations"""
        # Optimize thread hierarchy
        self.threadgroup_size = self._calculate_optimal_threadgroup_size()

        # Check for SIMD opportunities
        self.requires_simd_group = any(
            child.optimization_metadata['requires_simd_group']
            for child in self.children
        )

        # Optimize memory access
        for child in self.children:
            if isinstance(child, VariableNode):
                child._optimize_memory_access()

class VariableNode(CudaASTNode):
    """Variable declaration with enhanced Metal type mapping"""
    def __init__(self, name: str, data_type: str, qualifiers: List[str],
                 storage_class: Optional[str], initializer: Optional[List[CudaASTNode]] = None):
        super().__init__(kind='Variable', spelling=name)
        self.data_type = data_type
        self.qualifiers = qualifiers
        self.storage_class = storage_class
        self.initializer = initializer

        # Metal-specific attributes
        self.metal_type = self._map_to_metal_type()
        self.is_buffer = any(q in ['__global__', 'global'] for q in qualifiers)
        self.is_texture = 'texture' in data_type.lower()
        self.is_readonly = '__restrict__' in qualifiers
        self.metal_buffer_index = None

        if self.initializer:
            for init in self.initializer:
                self.add_child(init)

    def _map_to_metal_type(self) -> str:
        """Map CUDA type to Metal type"""
        base_type = self.data_type.replace('*', '').strip()
        return CUDA_TO_METAL_TYPE_MAP.get(base_type, base_type)

    def get_metal_type(self) -> str:
        """Get Metal type with qualifiers"""
        base_type = self.metal_type

        if '__shared__' in self.qualifiers:
            return f"threadgroup {base_type}"
        elif '__constant__' in self.qualifiers:
            return f"constant {base_type}"
        elif self.is_buffer:
            return f"device {base_type}"

        return base_type

    def is_buffer(self) -> bool:
        """Check if variable is a buffer"""
        return self.is_buffer

    def is_texture(self) -> bool:
        """Check if variable is a texture"""
        return self.is_texture

    def is_readonly(self) -> bool:
        """Check if variable is readonly"""
        return self.is_readonly

    def _generate_metal_translation(self) -> str:
        """Generate Metal variable declaration"""
        metal_type = self.get_metal_type()

        # Handle initialization
        init = ""
        if self.initializer:
            init_values = [init.get_metal_translation() for init in self.initializer]
            init = f" = {', '.join(init_values)}"

        # Handle array declarations
        if hasattr(self, 'array_dimensions') and self.array_dimensions:
            dims = ''.join(f'[{dim}]' for dim in self.array_dimensions)
            return f"{metal_type} {self.spelling}{dims}{init};"

        return f"{metal_type} {self.spelling}{init};"

class StatementNode(CudaASTNode):
    """Base class for all statement nodes"""
    def __init__(self, kind: str, spelling: Optional[str] = None):
        super().__init__(kind=kind, spelling=spelling)
        self.metal_scope: Optional[str] = None

class ExpressionNode(CudaASTNode):
    """Base class for all expression nodes"""
    def __init__(self, kind: str, spelling: Optional[str] = None, type: Optional[str] = None):
        super().__init__(kind=kind, spelling=spelling, type=type)
        self.result_type = type
        self.metal_expression: Optional[str] = None

class BinaryOperatorNode(ExpressionNode):
    """Binary operation with Metal operator mapping"""
    def __init__(self, operator: str, left: ExpressionNode, right: ExpressionNode):
        super().__init__(kind='BinaryOperator', spelling=operator)
        self.operator = operator
        self.left = left
        self.right = right
        self.metal_operator = CUDA_TO_METAL_OPERATORS.get(operator, operator)

        self.add_child(left)
        self.add_child(right)

    def _generate_metal_translation(self) -> str:
        left_trans = self.left.get_metal_translation()
        right_trans = self.right.get_metal_translation()

        # Handle special cases
        if self.operator == '/':
            if self.right.type in ['int', 'int32_t']:
                # Integer division needs explicit conversion in Metal
                return f"float({left_trans}) / float({right_trans})"

        return f"({left_trans} {self.metal_operator} {right_trans})"

class UnaryOperatorNode(ExpressionNode):
    """Unary operation with Metal operator mapping"""
    def __init__(self, operator: str, operand: ExpressionNode):
        super().__init__(kind='UnaryOperator', spelling=operator)
        self.operator = operator
        self.operand = operand
        self.metal_operator = CUDA_TO_METAL_OPERATORS.get(operator, operator)

        self.add_child(operand)

    def _generate_metal_translation(self) -> str:
        operand_trans = self.operand.get_metal_translation()
        return f"{self.metal_operator}({operand_trans})"

class CallExpressionNode(ExpressionNode):
    """Function call with Metal function mapping"""
    def __init__(self, function: ExpressionNode, arguments: List[ExpressionNode]):
        super().__init__(kind='CallExpression')
        self.function = function
        self.arguments = arguments

        self.add_child(function)
        for arg in arguments:
            self.add_child(arg)

    def _generate_metal_translation(self) -> str:
        func_name = self.function.spelling
        metal_func = CUDA_TO_METAL_FUNCTION_MAP.get(func_name, func_name)

        # Translate arguments
        metal_args = [arg.get_metal_translation() for arg in self.arguments]

        # Handle special cases
        if func_name in ['atomicAdd', 'atomicSub', 'atomicExch']:
            # Metal atomic functions need memory order
            metal_args.append('memory_order_relaxed')

        return f"{metal_func}({', '.join(metal_args)})"

class MemberAccessNode(ExpressionNode):
    """Member access expression with Metal support"""
    def __init__(self, base: ExpressionNode, member: str):
        super().__init__(kind='MemberAccess', spelling=member)
        self.base = base
        self.member = member
        self.add_child(base)

    def _generate_metal_translation(self) -> str:
        base_trans = self.base.get_metal_translation()

        # Handle CUDA built-in variables
        if isinstance(self.base, CudaBuiltinVariableNode):
            if self.base.spelling == 'threadIdx':
                return f"thread_position_in_threadgroup.{self.member}"
            elif self.base.spelling == 'blockIdx':
                return f"threadgroup_position.{self.member}"
            elif self.base.spelling == 'blockDim':
                return f"threads_per_threadgroup.{self.member}"
            elif self.base.spelling == 'gridDim':
                return f"threadgroups_per_grid.{self.member}"

        return f"{base_trans}.{self.member}"

class ArraySubscriptNode(ExpressionNode):
    """Array subscript with Metal optimization support"""
    def __init__(self, array: ExpressionNode, index: ExpressionNode):
        super().__init__(kind='ArraySubscript')
        self.array = array
        self.index = index
        self.add_child(array)
        self.add_child(index)

        # Optimization metadata
        self.optimization_metadata['coalesced_access'] = False
        self.optimization_metadata['vectorizable'] = False

    def _generate_metal_translation(self) -> str:
        array_trans = self.array.get_metal_translation()
        index_trans = self.index.get_metal_translation()

        if self.optimization_metadata['coalesced_access']:
            # Generate optimized access pattern
            return self._generate_coalesced_access(array_trans, index_trans)

        return f"{array_trans}[{index_trans}]"

    def _generate_coalesced_access(self, array: str, index: str) -> str:
        """Generate coalesced memory access pattern"""
        return f"{array}[{index} + thread_position_in_threadgroup.x]"

class CompoundStatementNode(StatementNode):
    """Block of statements"""
    def __init__(self, statements: List[Union[StatementNode, ExpressionNode]]):
        super().__init__(kind='CompoundStatement')
        self.statements = statements
        for stmt in statements:
            self.add_child(stmt)

    def _generate_metal_translation(self) -> str:
        metal_stmts = []
        for stmt in self.statements:
            trans = stmt.get_metal_translation()
            if trans:
                metal_stmts.extend(trans.split('\n'))
        return "{\n    " + "\n    ".join(metal_stmts) + "\n}"

class IfStatementNode(StatementNode):
    """If statement with Metal-specific optimizations"""
    def __init__(self, condition: ExpressionNode, then_branch: StatementNode,
                 else_branch: Optional[StatementNode] = None):
        super().__init__(kind='IfStatement')
        self.condition = condition
        self.then_branch = then_branch
        self.else_branch = else_branch

        self.add_child(condition)
        self.add_child(then_branch)
        if else_branch:
            self.add_child(else_branch)

    def _generate_metal_translation(self) -> str:
        cond_trans = self.condition.get_metal_translation()
        then_trans = self.then_branch.get_metal_translation()

        # Check if we can use select() for better performance
        if self._can_use_select():
            return self._generate_select_statement()

        result = f"if ({cond_trans}) {then_trans}"
        if self.else_branch:
            else_trans = self.else_branch.get_metal_translation()
            result += f" else {else_trans}"

        return result

    def _can_use_select(self) -> bool:
        """Check if we can use Metal's select function"""
        return (isinstance(self.then_branch, ExpressionNode) and
                isinstance(self.else_branch, ExpressionNode) and
                not self.optimization_metadata.get('requires_side_effects', False))

    def _generate_select_statement(self) -> str:
        """Generate Metal select statement"""
        cond = self.condition.get_metal_translation()
        then_expr = self.then_branch.get_metal_translation()
        else_expr = self.else_branch.get_metal_translation()
        return f"select({else_expr}, {then_expr}, {cond})"

class ForStatementNode(StatementNode):
    """For loop with Metal optimizations"""
    def __init__(self, init: Optional[Union[StatementNode, ExpressionNode]],
                 condition: Optional[ExpressionNode],
                 increment: Optional[ExpressionNode],
                 body: StatementNode):
        super().__init__(kind='ForStatement')
        self.init = init
        self.condition = condition
        self.increment = increment
        self.body = body

        if init:
            self.add_child(init)
        if condition:
            self.add_child(condition)
        if increment:
            self.add_child(increment)
        self.add_child(body)

        # Optimization metadata
        self.optimization_metadata.update({
            'unrollable': False,
            'vectorizable': False,
            'trip_count': None
        })

    def _generate_metal_translation(self) -> str:
        # Check for optimization opportunities
        if self._should_unroll():
            return self._generate_unrolled_loop()
        elif self._should_vectorize():
            return self._generate_vectorized_loop()

        # Standard for loop translation
        init_trans = self.init.get_metal_translation() if self.init else ""
        cond_trans = self.condition.get_metal_translation() if self.condition else "true"
        incr_trans = self.increment.get_metal_translation() if self.increment else ""
        body_trans = self.body.get_metal_translation()

        return f"for ({init_trans}; {cond_trans}; {incr_trans}) {body_trans}"

    def _should_unroll(self) -> bool:
        """Check if loop should be unrolled"""
        return (self.optimization_metadata['unrollable'] and
                self.optimization_metadata['trip_count'] is not None and
                self.optimization_metadata['trip_count'] <= 8)

    def _should_vectorize(self) -> bool:
        """Check if loop should be vectorized"""
        return (self.optimization_metadata['vectorizable'] and
                not self._has_loop_carried_dependencies())

    def _has_loop_carried_dependencies(self) -> bool:
        """Check for loop-carried dependencies"""
        # Implementation depends on analysis capabilities
        return False

    def _generate_unrolled_loop(self) -> str:
        """Generate unrolled loop"""
        trip_count = self.optimization_metadata['trip_count']
        body_trans = self.body.get_metal_translation()

        unrolled_stmts = []
        for i in range(trip_count):
            # Replace loop variable with constant
            stmt = body_trans.replace(self.init.spelling, str(i))
            unrolled_stmts.append(stmt)

        return "{\n    " + "\n    ".join(unrolled_stmts) + "\n}"

    def _generate_vectorized_loop(self) -> str:
        """Generate vectorized loop"""
        # Implementation depends on vectorization strategy
        return self._generate_metal_translation()

class ReturnStatementNode(StatementNode):
    """Return statement with Metal type compatibility"""
    def __init__(self, expression: Optional[ExpressionNode] = None):
        super().__init__(kind='ReturnStatement')
        self.expression = expression
        if expression:
            self.add_child(expression)

    def _generate_metal_translation(self) -> str:
        if not self.expression:
            return "return;"
        expr_trans = self.expression.get_metal_translation()
        return f"return {expr_trans};"

class CastExpressionNode(ExpressionNode):
    """Type cast with Metal type mapping"""
    def __init__(self, target_type: str, expression: ExpressionNode):
        super().__init__(kind='CastExpression', type=target_type)
        self.target_type = target_type
        self.expression = expression
        self.add_child(expression)

    def _generate_metal_translation(self) -> str:
        metal_type = CUDA_TO_METAL_TYPE_MAP.get(self.target_type, self.target_type)
        expr_trans = self.expression.get_metal_translation()
        return f"({metal_type})({expr_trans})"

class CudaSharedMemoryNode(CudaASTNode):
    """CUDA shared memory with Metal threadgroup translation"""
    def __init__(self, variable: VariableNode):
        super().__init__(kind='CudaSharedMemory')
        self.variable = variable
        self.metal_type = 'threadgroup'
        self.add_child(variable)

    def _generate_metal_translation(self) -> str:
        var_trans = self.variable.get_metal_translation()
        return f"threadgroup {var_trans}"

class CudaAtomicOperationNode(CudaASTNode):
    """CUDA atomic operations with Metal atomic translation"""
    def __init__(self, operation: str, arguments: List[ExpressionNode]):
        super().__init__(kind='CudaAtomicOperation', spelling=operation)
        self.operation = operation
        self.arguments = arguments
        for arg in arguments:
            self.add_child(arg)

    def _generate_metal_translation(self) -> str:
        metal_op = CUDA_TO_METAL_FUNCTION_MAP.get(self.operation)
        if not metal_op:
            raise ValueError(f"Unsupported atomic operation: {self.operation}")

        args_trans = [arg.get_metal_translation() for arg in self.arguments]
        # Metal atomic operations require memory order
        args_trans.append("memory_order_relaxed")
        return f"{metal_op}({', '.join(args_trans)})"

class CudaTextureNode(CudaASTNode):
    """CUDA texture with Metal texture translation"""
    def __init__(self, name: str, dimensions: int, type: str):
        super().__init__(kind='CudaTexture', spelling=name)
        self.dimensions = dimensions
        self.type = type
        self.metal_texture_type = self._get_metal_texture_type()

    def _get_metal_texture_type(self) -> str:
        if self.dimensions == 1:
            return "texture1d"
        elif self.dimensions == 2:
            return "texture2d"
        elif self.dimensions == 3:
            return "texture3d"
        raise ValueError(f"Unsupported texture dimensions: {self.dimensions}")

    def _generate_metal_translation(self) -> str:
        base_type = CUDA_TO_METAL_TYPE_MAP.get(self.type, 'float')
        return f"{self.metal_texture_type}<{base_type}>"

class CudaMallocNode(CudaASTNode):
    """CUDA memory allocation with Metal buffer creation"""
    def __init__(self, devPtr: ExpressionNode, size: ExpressionNode):
        super().__init__(kind='CudaMalloc')
        self.devPtr = devPtr
        self.size = size
        self.add_child(devPtr)
        self.add_child(size)

    def _generate_metal_translation(self) -> str:
        ptr_trans = self.devPtr.get_metal_translation()
        size_trans = self.size.get_metal_translation()
        return f"device.makeBuffer(length: {size_trans}, options: MTLResourceOptions.storageModeShared)"

class CudaMemcpyNode(CudaASTNode):
    """CUDA memcpy with Metal buffer copy"""
    def __init__(self, dst: ExpressionNode, src: ExpressionNode, count: ExpressionNode, kind: str):
        super().__init__(kind='CudaMemcpy')
        self.dst = dst
        self.src = src
        self.count = count
        self.kind = kind
        self.add_child(dst)
        self.add_child(src)
        self.add_child(count)

    def _generate_metal_translation(self) -> str:
        dst_trans = self.dst.get_metal_translation()
        src_trans = self.src.get_metal_translation()
        count_trans = self.count.get_metal_translation()

        if self.kind == 'cudaMemcpyHostToDevice':
            return f"memcpy({dst_trans}.contents, {src_trans}, {count_trans})"
        elif self.kind == 'cudaMemcpyDeviceToHost':
            return f"memcpy({dst_trans}, {src_trans}.contents, {count_trans})"
        elif self.kind == 'cudaMemcpyDeviceToDevice':
            return f"commandBuffer.copy(from: {src_trans}, to: {dst_trans}, size: {count_trans})"
        else:
            raise ValueError(f"Unsupported memcpy kind: {self.kind}")

class CudaSyncthreadsNode(CudaASTNode):
    """CUDA syncthreads with Metal barrier"""
    def __init__(self):
        super().__init__(kind='CudaSyncthreads')

    def _generate_metal_translation(self) -> str:
        return "threadgroup_barrier(mem_flags::mem_threadgroup)"

class CudaEventNode(CudaASTNode):
    """Base class for CUDA event operations"""
    def __init__(self, kind: str, event: ExpressionNode):
        super().__init__(kind=kind)
        self.event = event
        self.add_child(event)

class CudaEventCreateNode(CudaEventNode):
    def __init__(self, event: ExpressionNode):
        super().__init__('CudaEventCreate', event)

    def _generate_metal_translation(self) -> str:
        event_trans = self.event.get_metal_translation()
        return f"{event_trans} = device.makeEvent()"

class CudaEventRecordNode(CudaEventNode):
    def __init__(self, event: ExpressionNode, stream: Optional[ExpressionNode] = None):
        super().__init__('CudaEventRecord', event)
        self.stream = stream
        if stream:
            self.add_child(stream)

    def _generate_metal_translation(self) -> str:
        event_trans = self.event.get_metal_translation()
        if self.stream:
            stream_trans = self.stream.get_metal_translation()
            return f"{stream_trans}.encodeSignalEvent({event_trans})"
        return f"commandBuffer.encodeSignalEvent({event_trans})"

class CudaEventSynchronizeNode(CudaEventNode):
    def __init__(self, event: ExpressionNode):
        super().__init__('CudaEventSynchronize', event)

    def _generate_metal_translation(self) -> str:
        event_trans = self.event.get_metal_translation()
        return f"{event_trans}.wait()"

# Optimization utilities for Metal GPU code generation
from typing import List, Dict, Any, Optional, Set, Tuple
import math
from enum import Enum

class AccessPattern(Enum):
    SEQUENTIAL = "sequential"
    STRIDED = "strided"
    RANDOM = "random"
    COALESCED = "coalesced"

class OptimizationLevel(Enum):
    NONE = 0
    BASIC = 1
    AGGRESSIVE = 2
    MAXIMUM = 3

class MetalOptimizer:
    """Core optimization utilities for Metal GPU code"""

    def __init__(self, optimization_level: OptimizationLevel = OptimizationLevel.BASIC):
        self.optimization_level = optimization_level
        self.simd_width = 32  # Metal GPU SIMD width
        self.max_threads_per_threadgroup = 1024
        self.max_total_threadgroup_memory = 32768  # 32KB
        self.preferred_workgroup_multiple = 32

    def optimize_threadgroup_size(self, requested_size: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Optimize threadgroup size for Metal GPU execution"""
        x, y, z = requested_size
        total_threads = x * y * z

        if total_threads > self.max_threads_per_threadgroup:
            # Scale down dimensions while maintaining ratios
            scale = math.sqrt(self.max_threads_per_threadgroup / total_threads)
            x = min(int(x * scale), self.max_threads_per_threadgroup)
            y = min(int(y * scale), self.max_threads_per_threadgroup // x)
            z = min(int(z * scale), self.max_threads_per_threadgroup // (x * y))

        # Ensure x dimension is multiple of SIMD width
        x = ((x + self.simd_width - 1) // self.simd_width) * self.simd_width

        return (x, y, z)

    def analyze_memory_access(self, access_pattern: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze memory access patterns for optimization"""
        result = {
            'pattern': AccessPattern.RANDOM,
            'stride': None,
            'coalescing_opportunity': False,
            'vectorization_opportunity': False,
            'cache_locality': 0.0,
            'bank_conflicts': [],
        }

        # Detect sequential access
        if self._is_sequential_access(access_pattern):
            result['pattern'] = AccessPattern.SEQUENTIAL
            result['coalescing_opportunity'] = True
            result['cache_locality'] = 1.0

        # Detect strided access
        elif (stride := self._detect_stride(access_pattern)) is not None:
            result['pattern'] = AccessPattern.STRIDED
            result['stride'] = stride
            result['vectorization_opportunity'] = self._can_vectorize(stride)
            result['cache_locality'] = 1.0 / stride

        # Check for bank conflicts in threadgroup memory
        if bank_conflicts := self._check_bank_conflicts(access_pattern):
            result['bank_conflicts'] = bank_conflicts

        return result

    def optimize_kernel_launch(self, grid_size: Tuple[int, int, int],
                               block_size: Tuple[int, int, int]) -> Dict[str, Any]:
        """Optimize kernel launch configuration for Metal"""
        optimized_block = self.optimize_threadgroup_size(block_size)

        # Calculate grid size based on optimized block size
        optimized_grid = tuple(
            (grid_size[i] * block_size[i] + optimized_block[i] - 1) // optimized_block[i]
            for i in range(3)
        )

        return {
            'threadgroup_size': optimized_block,
            'grid_size': optimized_grid,
            'simd_groups_per_threadgroup': optimized_block[0] // self.simd_width,
            'thread_execution_width': self.simd_width,
            'max_total_threads': optimized_grid[0] * optimized_grid[1] * optimized_grid[2] *
                                 optimized_block[0] * optimized_block[1] * optimized_block[2]
        }

    def optimize_memory_layout(self, buffer_size: int, access_info: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize memory layout for Metal buffers"""
        alignment = 16  # Base alignment for Metal buffers
        padded_size = (buffer_size + alignment - 1) & ~(alignment - 1)

        result = {
            'size': padded_size,
            'alignment': alignment,
            'padding': padded_size - buffer_size,
            'layout_strategy': 'linear'
        }

        # Apply advanced optimizations based on access pattern
        if access_info['pattern'] == AccessPattern.SEQUENTIAL:
            result['layout_strategy'] = 'sequential_optimized'
            result['prefetch_distance'] = self.simd_width * 4

        elif access_info['pattern'] == AccessPattern.STRIDED:
            if access_info['vectorization_opportunity']:
                result['layout_strategy'] = 'vectorized'
                result['vector_width'] = min(4, access_info['stride'])
                alignment = max(alignment, result['vector_width'] * 4)

        return result

    def generate_barrier_optimization(self, sync_points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimize barrier placement and type"""
        optimized_barriers = []

        for sync_point in sync_points:
            if self._can_remove_barrier(sync_point):
                continue

            barrier_type = self._select_optimal_barrier(sync_point)
            optimized_barriers.append({
                'location': sync_point['location'],
                'type': barrier_type,
                'scope': sync_point.get('scope', 'threadgroup'),
                'optimization_applied': True
            })

        return optimized_barriers

    def optimize_arithmetic_operations(self, operations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimize arithmetic operations for Metal"""
        optimized_ops = []

        for op in operations:
            if op['type'] == 'binary':
                opt_op = self._optimize_binary_operation(op)
            elif op['type'] == 'unary':
                opt_op = self._optimize_unary_operation(op)
            elif op['type'] == 'math_function':
                opt_op = self._optimize_math_function(op)
            else:
                opt_op = op

            if fast_variant := self._get_fast_math_variant(opt_op):
                opt_op['implementation'] = fast_variant

            optimized_ops.append(opt_op)

        return optimized_ops

    # Private helper methods
    def _is_sequential_access(self, access_pattern: List[Dict[str, Any]]) -> bool:
        """Check if memory access pattern is sequential"""
        if not access_pattern:
            return False

        expected_idx = access_pattern[0]['index']
        for access in access_pattern[1:]:
            if access['index'] != expected_idx + 1:
                return False
            expected_idx = access['index']

        return True

    def _detect_stride(self, access_pattern: List[Dict[str, Any]]) -> Optional[int]:
        """Detect strided access pattern"""
        if len(access_pattern) < 2:
            return None

        stride = access_pattern[1]['index'] - access_pattern[0]['index']
        for i in range(1, len(access_pattern) - 1):
            if access_pattern[i + 1]['index'] - access_pattern[i]['index'] != stride:
                return None

        return stride

    def _can_vectorize(self, stride: int) -> bool:
        """Check if access pattern can be vectorized"""
        return (stride in (2, 4, 8, 16) and
                self.optimization_level >= OptimizationLevel.BASIC)

    def _check_bank_conflicts(self, access_pattern: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect bank conflicts in threadgroup memory access"""
        conflicts = []
        bank_accesses = {}

        for access in access_pattern:
            bank = access['index'] % 32  # Metal uses 32 banks
            if bank in bank_accesses:
                conflicts.append({
                    'bank': bank,
                    'first_access': bank_accesses[bank],
                    'conflicting_access': access
                })
            bank_accesses[bank] = access

        return conflicts

    def _can_remove_barrier(self, sync_point: Dict[str, Any]) -> bool:
        """Check if barrier can be safely removed"""
        return (not sync_point.get('required_by_dependency', True) and
                self.optimization_level >= OptimizationLevel.AGGRESSIVE)

    def _select_optimal_barrier(self, sync_point: Dict[str, Any]) -> str:
        """Select optimal barrier type based on synchronization requirements"""
        if sync_point.get('scope') == 'device':
            return 'device_memory_barrier'
        elif sync_point.get('scope') == 'threadgroup':
            if sync_point.get('memory_access_only', False):
                return 'threadgroup_memory_barrier'
            else:
                return 'threadgroup_barrier'
        return 'threadgroup_barrier'

    def _get_fast_math_variant(self, operation: Dict[str, Any]) -> Optional[str]:
        """Get fast math variant of operation if available"""
        fast_variants = {
            'sin': 'fast::sin',
            'cos': 'fast::cos',
            'exp': 'fast::exp',
            'log': 'fast::log',
            'pow': 'fast::pow',
            'rsqrt': 'fast::rsqrt'
        }

        if (operation['type'] == 'math_function' and
                operation['function'] in fast_variants and
                self.optimization_level >= OptimizationLevel.BASIC):
            return fast_variants[operation['function']]

        return None

    def _optimize_binary_operation(self, op: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize binary operation"""
        optimized = op.copy()

        # Strength reduction
        if op['operator'] == '*' and self._is_power_of_two(op.get('right')):
            optimized['operator'] = '<<'
            optimized['right'] = self._log2(op['right'])

        # Vector operation opportunities
        elif self._can_vectorize_operation(op):
            optimized['vectorized'] = True
            optimized['vector_width'] = 4

        return optimized

    def _optimize_unary_operation(self, op: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize unary operation"""
        optimized = op.copy()

        if op['operator'] == '-' and self._can_fuse_with_next(op):
            optimized['fused_with_next'] = True

        return optimized

    def _optimize_math_function(self, op: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize math function"""
        optimized = op.copy()

        if self.optimization_level >= OptimizationLevel.BASIC:
            optimized['use_fast_math'] = True

        if op['function'] == 'pow' and self._is_constant_integer(op.get('exponent')):
            optimized['unrolled'] = True

        return optimized

    @staticmethod
    def _is_power_of_two(n: Any) -> bool:
        """Check if a number is a power of two"""
        if not isinstance(n, (int, float)):
            return False
        return n > 0 and (n & (n - 1)) == 0

    @staticmethod
    def _log2(n: int) -> int:
        """Calculate integer log2"""
        return n.bit_length() - 1

    def _can_vectorize_operation(self, op: Dict[str, Any]) -> bool:
        """Check if operation can be vectorized"""
        return (self.optimization_level >= OptimizationLevel.BASIC and
                op.get('data_type') in ('float', 'int32_t', 'uint32_t') and
                not op.get('has_side_effects', False))

    def _can_fuse_with_next(self, op: Dict[str, Any]) -> bool:
        """Check if operation can be fused with next operation"""
        return (self.optimization_level >= OptimizationLevel.AGGRESSIVE and
                not op.get('has_side_effects', False))

    @staticmethod
    def _is_constant_integer(value: Any) -> bool:
        """Check if value is a constant integer"""
        return isinstance(value, int)