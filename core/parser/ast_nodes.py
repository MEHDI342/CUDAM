from __future__ import annotations  # Allows forward references in type hints
import re
import sys
from typing import Dict, List, Optional, Union, Any, Set
from dataclasses import dataclass
from enum import Enum, auto


class CUDAType(Enum):
    """CUDA built-in types following NVIDIA specification"""
    VOID = "void"
    CHAR = "char"
    UCHAR = "unsigned char"
    SHORT = "short"
    USHORT = "unsigned short"
    INT = "int"
    UINT = "unsigned int"
    LONG = "long"
    ULONG = "unsigned long"
    FLOAT = "float"
    DOUBLE = "double"
    DIM3 = "dim3"
    SIZE_T = "size_t"
    CUDAERROR = "cudaError_t"

    # Vector types
    CHAR1 = "char1"
    CHAR2 = "char2"
    CHAR3 = "char3"
    CHAR4 = "char4"
    UCHAR1 = "uchar1"
    UCHAR2 = "uchar2"
    UCHAR3 = "uchar3"
    UCHAR4 = "uchar4"
    SHORT1 = "short1"
    SHORT2 = "short2"
    SHORT3 = "short3"
    SHORT4 = "short4"
    USHORT1 = "ushort1"
    USHORT2 = "ushort2"
    USHORT3 = "ushort3"
    USHORT4 = "ushort4"
    INT1 = "int1"
    INT2 = "int2"
    INT3 = "int3"
    INT4 = "int4"
    UINT1 = "uint1"
    UINT2 = "uint2"
    UINT3 = "uint3"
    UINT4 = "uint4"
    LONG1 = "long1"
    LONG2 = "long2"
    LONG3 = "long3"
    LONG4 = "long4"
    ULONG1 = "ulong1"
    ULONG2 = "ulong2"
    ULONG3 = "ulong3"
    ULONG4 = "ulong4"
    FLOAT1 = "float1"
    FLOAT2 = "float2"
    FLOAT3 = "float3"
    FLOAT4 = "float4"
    DOUBLE1 = "double1"
    DOUBLE2 = "double2"
    DOUBLE3 = "double3"
    DOUBLE4 = "double4"

    @classmethod
    def is_vector_type(cls, type_name: str) -> bool:
        return any(v.value == type_name for v in cls) and any(str(i) in type_name for i in range(1, 5))


class CUDAQualifier(Enum):
    """CUDA type qualifiers following NVIDIA specification"""
    CONST = "__const__"
    DEVICE = "__device__"
    GLOBAL = "__global__"
    HOST = "__host__"
    LOCAL = "__local__"
    SHARED = "__shared__"
    RESTRICT = "__restrict__"
    MANAGED = "__managed__"


@dataclass
class SourceLocation:
    """Source code location information."""
    file: str
    line: int
    column: int
    offset: int


class CUDANodeType(Enum):
    """Enumeration of all CUDA AST node types."""
    # ... (keep existing types) ...
    COMPOUND_STMT = auto()
    TEXTURE = auto()
    BARRIER = auto()
    # Add other node types as needed


class CUDANode:
    """Base class for all CUDA AST nodes"""
    def __init__(self, line: int, column: int):
        self.line = line
        self.column = column
        self.children: List[CUDANode] = []
        self.parent: Optional[CUDANode] = None
        self.cuda_type: Optional[CUDAType] = None
        self.qualifiers: Set[CUDAQualifier] = set()

    def add_child(self, node: CUDANode):
        self.children.append(node)
        node.parent = self
        return node

    def add_qualifier(self, qualifier: CUDAQualifier):
        self.qualifiers.add(qualifier)

    def is_kernel(self) -> bool:
        return CUDAQualifier.GLOBAL in self.qualifiers

    def is_device_func(self) -> bool:
        return CUDAQualifier.DEVICE in self.qualifiers


class CUDACompoundStmt(CUDANode):
    """Represents a compound statement (block of code)."""
    def __init__(self,
                 statements: List[CUDANode],
                 line: int,
                 column: int):
        super().__init__(line, column)
        self.node_type = CUDANodeType.COMPOUND_STMT
        for stmt in statements:
            self.add_child(stmt)

    def get_statements(self) -> List[CUDANode]:
        """Get all statements in this compound statement."""
        return self.children


class CUDATexture(CUDANode):
    """Represents a CUDA texture declaration."""
    def __init__(self,
                 name: str,
                 texture_type: str,
                 dimensions: int,
                 is_readonly: bool,
                 line: int,
                 column: int):
        super().__init__(line, column)
        self.node_type = CUDANodeType.TEXTURE
        self.name = name
        self.texture_type = texture_type
        self.dimensions = dimensions
        self.is_readonly = is_readonly
        self.normalized_coords = False
        self.filter_mode = "point"  # or "linear"
        self.address_mode = "clamp"  # or "wrap", "mirror", "border"

    def set_texture_options(self,
                            normalized_coords: bool = False,
                            filter_mode: str = "point",
                            address_mode: str = "clamp"):
        """Set texture sampling options."""
        self.normalized_coords = normalized_coords
        self.filter_mode = filter_mode
        self.address_mode = address_mode


class CUDABarrier(CUDANode):
    """Represents a CUDA synchronization barrier."""
    BARRIER_TYPES = {
        'THREADS': '__syncthreads',
        'DEVICE': '__threadfence',
        'BLOCK': '__threadfence_block',
        'SYSTEM': '__threadfence_system'
    }

    def __init__(self,
                 barrier_type: str,
                 line: int,
                 column: int):
        super().__init__(line, column)
        self.node_type = CUDANodeType.BARRIER
        if barrier_type not in self.BARRIER_TYPES:
            raise ValueError(f"Invalid barrier type: {barrier_type}")
        self.barrier_type = barrier_type
        self.barrier_function = self.BARRIER_TYPES[barrier_type]

    def is_thread_sync(self) -> bool:
        """Check if this is a thread synchronization barrier."""
        return self.barrier_type == 'THREADS'

    def is_memory_fence(self) -> bool:
        """Check if this is a memory fence operation."""
        return self.barrier_type in ['DEVICE', 'BLOCK', 'SYSTEM']


class CUDAExpressionNode(CUDANode):
    """Base class for CUDA expressions."""
    def __init__(self,
                 expression_type: CUDAType,
                 line: int,
                 column: int):
        super().__init__(line, column)
        self.cuda_type = expression_type
        self.result_type = expression_type
        self.is_lvalue = False
        self.is_constant = False

    def get_type(self) -> CUDAType:
        """Get the CUDA type of this expression."""
        return self.cuda_type

    def is_assignable(self) -> bool:
        """Check if expression can be assigned to."""
        return self.is_lvalue


class CUDAStatement(CUDANode):
    """Base class for CUDA statements."""
    def __init__(self,
                 node_type: CUDANodeType,
                 line: int,
                 column: int):
        super().__init__(line, column)
        self.node_type = node_type
        self.scope_level = 0
        self.has_side_effects = False

    def get_scope_level(self) -> int:
        """Get the scope nesting level of this statement."""
        return self.scope_level

    def set_scope_level(self, level: int):
        """Set the scope nesting level."""
        self.scope_level = level

    def has_control_flow(self) -> bool:
        """Check if statement affects control flow."""
        return False


class CUDAKernel(CUDANode):
    """CUDA kernel function definition"""
    def __init__(self,
                 name: str,
                 return_type: CUDAType,
                 parameters: List[CUDAParameter],
                 line: int,
                 column: int):
        super().__init__(line, column)
        self.name = name
        self.return_type = return_type
        self.parameters = parameters
        self.launch_bounds: Optional[Dict[str, int]] = None
        self.add_qualifier(CUDAQualifier.GLOBAL)

        # Add parameters as children
        for param in parameters:
            self.add_child(param)

    def set_launch_bounds(self, max_threads: int, min_blocks: Optional[int] = None):
        self.launch_bounds = {
            'maxThreadsPerBlock': max_threads
        }
        if min_blocks is not None:
            self.launch_bounds['minBlocksPerMultiprocessor'] = min_blocks


class CUDAParameter(CUDANode):
    """CUDA kernel parameter"""
    def __init__(self,
                 name: str,
                 param_type: CUDAType,
                 is_pointer: bool,
                 line: int,
                 column: int):
        super().__init__(line, column)
        self.name = name
        self.cuda_type = param_type
        self.is_pointer = is_pointer


class CUDASharedMemory(CUDANode):
    """CUDA shared memory declaration"""
    def __init__(self,
                 name: str,
                 data_type: CUDAType,
                 size: Optional[int],
                 line: int,
                 column: int):
        super().__init__(line, column)
        self.name = name
        self.cuda_type = data_type
        self.size = size
        self.add_qualifier(CUDAQualifier.SHARED)


class CUDAThreadIdx(CUDANode):
    """CUDA thread index access (threadIdx)"""
    def __init__(self, dimension: str, line: int, column: int):
        super().__init__(line, column)
        if dimension not in ['x', 'y', 'z']:
            raise ValueError(f"Invalid thread dimension: {dimension}")
        self.dimension = dimension
        self.cuda_type = CUDAType.UINT


class CUDABlockIdx(CUDANode):
    """CUDA block index access (blockIdx)"""
    def __init__(self, dimension: str, line: int, column: int):
        super().__init__(line, column)
        if dimension not in ['x', 'y', 'z']:
            raise ValueError(f"Invalid block dimension: {dimension}")
        self.dimension = dimension
        self.cuda_type = CUDAType.UINT


class CUDAGridDim(CUDANode):
    """CUDA grid dimension access (gridDim)"""
    def __init__(self, dimension: str, line: int, column: int):
        super().__init__(line, column)
        if dimension not in ['x', 'y', 'z']:
            raise ValueError(f"Invalid grid dimension: {dimension}")
        self.dimension = dimension
        self.cuda_type = CUDAType.UINT


class CUDAAtomicOperation(CUDANode):
    """CUDA atomic operation"""
    VALID_OPS = {'Add', 'Sub', 'Exch', 'Min', 'Max', 'Inc', 'Dec', 'CAS',
                 'And', 'Or', 'Xor'}

    def __init__(self, operation: str, line: int, column: int):
        super().__init__(line, column)
        if operation not in self.VALID_OPS:
            raise ValueError(f"Invalid atomic operation: {operation}")
        self.operation = operation


class CUDASync(CUDANode):
    """CUDA synchronization primitives"""
    SYNC_TYPES = {
        'syncthreads': '__syncthreads',
        'threadfence': '__threadfence',
        'threadfence_block': '__threadfence_block',
        'threadfence_system': '__threadfence_system'
    }

    def __init__(self, sync_type: str, line: int, column: int):
        super().__init__(line, column)
        if sync_type not in self.SYNC_TYPES:
            raise ValueError(f"Invalid sync type: {sync_type}")
        self.sync_type = self.SYNC_TYPES[sync_type]


# Alias assignments moved after class definitions
KernelNode = CUDAKernel
ParameterNode = CUDAParameter
CompoundStmtNode = CUDACompoundStmt
TextureNode = CUDATexture
BarrierNode = CUDABarrier
