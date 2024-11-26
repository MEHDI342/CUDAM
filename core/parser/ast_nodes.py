"""
Core AST node definitions for CUDA-to-Metal translation.
Provides comprehensive type system and node hierarchy for robust AST manipulation.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Union, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto


# Define CUDA Types
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
        """Check if the type is a vector type."""
        return any(v.value == type_name for v in cls) and any(str(i) in type_name for i in range(1, 5))


# Define CUDA Qualifiers
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


# Define CUDA AST Node Types
class CUDANodeType(Enum):
    """Enumeration of all CUDA AST node types"""
    COMPOUND_STMT = auto()
    TEXTURE = auto()
    BARRIER = auto()
    ATOMIC = auto()
    THREAD_IDX = auto()
    BLOCK_IDX = auto()
    GRID_DIM = auto()
    BLOCK_DIM = auto()
    KERNEL = auto()
    FUNCTION = auto()
    VARIABLE = auto()
    PARAMETER = auto()
    STRUCT = auto()
    CLASS = auto()
    ENUM = auto()
    TYPEDEF = auto()
    NAMESPACE = auto()
    TEMPLATE = auto()


# Base CUDA AST Node
class CUDANode:
    """Base class for all CUDA AST nodes"""
    def __init__(self, line: int, column: int):
        self.line = line
        self.column = column
        self.children: List[CUDANode] = []
        self.parent: Optional[CUDANode] = None
        self.cuda_type: Optional[CUDAType] = None
        self.qualifiers: Set[CUDAQualifier] = set()
        self.metal_translation: Optional[str] = None
        self.optimization_hints: Dict[str, Any] = {}

    def add_child(self, node: CUDANode) -> CUDANode:
        """Add child node with validation"""
        self.children.append(node)
        node.parent = self
        return node

    def add_qualifier(self, qualifier: CUDAQualifier) -> None:
        """Add type qualifier with validation"""
        self.qualifiers.add(qualifier)

    def is_kernel(self) -> bool:
        """Check if node represents a CUDA kernel"""
        return CUDAQualifier.GLOBAL in self.qualifiers

    def is_device_func(self) -> bool:
        """Check if node represents a device function"""
        return CUDAQualifier.DEVICE in self.qualifiers

    def traverse(self, callback: callable) -> None:
        """Traverse AST applying callback to each node"""
        callback(self)
        for child in self.children:
            child.traverse(callback)


# Source Location Information
@dataclass
class SourceLocation:
    """Source code location information"""
    file: str
    line: int
    column: int
    offset: int


# CUDA Expression Node
@dataclass
class CUDAExpressionNode(CUDANode):
    """Represents an expression in CUDA code"""
    kind: str
    operator: Optional[str] = None
    left: Optional[CUDAExpressionNode] = None
    right: Optional[CUDAExpressionNode] = None
    operand: Optional[CUDAExpressionNode] = None
    function: Optional[str] = None
    arguments: List[CUDAExpressionNode] = field(default_factory=list)
    array: Optional[CUDAExpressionNode] = None
    index: Optional[CUDAExpressionNode] = None
    base: Optional[CUDAExpressionNode] = None
    member: Optional[str] = None
    condition: Optional[CUDAExpressionNode] = None
    then_expression: Optional[CUDAExpressionNode] = None
    else_expression: Optional[CUDAExpressionNode] = None
    elements: List[CUDAExpressionNode] = field(default_factory=list)
    spelling: Optional[str] = None


# CUDA Statement Node
@dataclass
class CUDAStatement(CUDANode):
    """Represents a statement in CUDA code"""
    kind: str
    expression: Optional[CUDAExpressionNode] = None
    condition: Optional[CUDAExpressionNode] = None
    then_branch: List[CUDANode] = field(default_factory=list)
    else_branch: List[CUDANode] = field(default_factory=list)
    init: Optional[CUDAExpressionNode] = None
    increment: Optional[CUDAExpressionNode] = None
    body: List[CUDANode] = field(default_factory=list)


# Function Base Node
class FunctionNode(CUDANode):
    """Base class for functions"""
    def __init__(self, name: str, return_type: CUDAType,
                 parameters: List[CUDAParameter], body: CUDACompoundStmt,
                 line: int, column: int):
        super().__init__(line, column)
        self.name = name
        self.return_type = return_type
        self.parameters = parameters
        self.body = body
        self.metal_name: Optional[str] = None

    def add_parameter(self, parameter: CUDAParameter) -> None:
        """Add a parameter to the function"""
        self.parameters.append(parameter)
        self.add_child(parameter)


# Kernel Node
class CUDAKernel(FunctionNode):
    """CUDA kernel function definition"""
    def __init__(self, name: str, return_type: CUDAType,
                 parameters: List[CUDAParameter], body: CUDACompoundStmt,
                 line: int, column: int):
        super().__init__(name, return_type, parameters, body, line, column)
        self.add_qualifier(CUDAQualifier.GLOBAL)
        self.launch_bounds: Optional[Dict[str, int]] = None
        self.thread_hierarchy: Dict[str, Dict[str, int]] = {
            'block': {'x': 256, 'y': 1, 'z': 1},
            'grid': {'x': 1, 'y': 1, 'z': 1}
        }
        self.shared_memory_size = 0
        self.shared_memory_vars: List[CUDASharedMemory] = []

    def set_launch_bounds(self, max_threads: int, min_blocks: Optional[int] = None) -> None:
        """Set kernel launch bounds with validation"""
        self.launch_bounds = {
            'maxThreadsPerBlock': max_threads
        }
        if min_blocks is not None:
            self.launch_bounds['minBlocksPerMultiprocessor'] = min_blocks


# CUDA Parameter Node
class CUDAParameter(CUDANode):
    """CUDA kernel parameter"""
    def __init__(self, name: str, param_type: CUDAType, is_pointer: bool,
                 line: int, column: int):
        super().__init__(line, column)
        self.name = name
        self.param_type = param_type
        self.is_pointer = is_pointer
        self.array_dims: List[int] = []

    def add_array_dimension(self, size: int) -> None:
        """Add array dimension with validation"""
        self.array_dims.append(size)


# CUDA Shared Memory Declaration
class CUDASharedMemory(CUDANode):
    """CUDA shared memory declaration"""
    def __init__(self, name: str, data_type: CUDAType, size: Optional[int],
                 line: int, column: int):
        super().__init__(line, column)
        self.name = name
        self.data_type = data_type
        self.size = size
        self.is_dynamic = size is None
        self.add_qualifier(CUDAQualifier.SHARED)


# CUDA Thread Index Access
class CUDAThreadIdx(CUDANode):
    """CUDA thread index access (threadIdx)"""
    def __init__(self, dimension: str, line: int, column: int):
        super().__init__(line, column)
        if dimension not in ['x', 'y', 'z']:
            raise ValueError(f"Invalid thread dimension: {dimension}")
        self.dimension = dimension
        self.cuda_type = CUDAType.UINT


# CUDA Block Index Access
class CUDABlockIdx(CUDANode):
    """CUDA block index access (blockIdx)"""
    def __init__(self, dimension: str, line: int, column: int):
        super().__init__(line, column)
        if dimension not in ['x', 'y', 'z']:
            raise ValueError(f"Invalid block dimension: {dimension}")
        self.dimension = dimension
        self.cuda_type = CUDAType.UINT


# CUDA Compound Statement
class CUDACompoundStmt(CUDANode):
    """Represents a compound statement (block of code)"""
    def __init__(self, statements: List[CUDANode], line: int, column: int):
        super().__init__(line, column)
        self.node_type = CUDANodeType.COMPOUND_STMT
        for stmt in statements:
            self.add_child(stmt)

    def get_statements(self) -> List[CUDANode]:
        """Get all statements in compound statement"""
        return self.children


# Variable Declaration Node
class VariableNode(CUDANode):
    """Variable declaration node"""
    def __init__(self, name: str, var_type: CUDAType, is_pointer: bool,
                 line: int, column: int):
        super().__init__(line, column)
        self.name = name
        self.var_type = var_type
        self.is_pointer = is_pointer
        self.is_array = False
        self.array_dims: List[int] = []
        self.initializer: Optional[CUDANode] = None

    def add_array_dimension(self, size: int) -> None:
        """Add array dimension with validation"""
        self.array_dims.append(size)
        self.is_array = True

    def set_initializer(self, initializer: CUDANode) -> None:
        """Set initializer for the variable"""
        self.initializer = initializer
        self.add_child(initializer)


# Structure Definition Node
class StructNode(CUDANode):
    """Structure definition node"""
    def __init__(self, name: str, fields: List[VariableNode],
                 line: int, column: int):
        super().__init__(line, column)
        self.name = name
        self.fields = fields
        self.is_packed = False
        for field in fields:
            self.add_child(field)


# Enumeration Definition Node
class EnumNode(CUDANode):
    """Enumeration definition node"""
    def __init__(self, name: str, values: Dict[str, int],
                 line: int, column: int):
        super().__init__(line, column)
        self.name = name
        self.values = values
        self.underlying_type = CUDAType.INT


# Typedef Definition Node
class TypedefNode(CUDANode):
    """Typedef definition node"""
    def __init__(self, name: str, original_type: CUDAType,
                 line: int, column: int):
        super().__init__(line, column)
        self.name = name
        self.original_type = original_type
        self.metal_type: Optional[str] = None


# Class Definition Node
class ClassNode(CUDANode):
    """Class definition node"""
    def __init__(self, name: str, methods: List[FunctionNode],
                 fields: List[VariableNode], line: int, column: int):
        super().__init__(line, column)
        self.name = name
        self.methods = methods
        self.fields = fields
        self.base_classes: List[str] = []
        for method in methods:
            self.add_child(method)
        for field in fields:
            self.add_child(field)


# Namespace Definition Node
class NamespaceNode(CUDANode):
    """Namespace definition node"""
    def __init__(self, name: str, declarations: List[CUDANode],
                 line: int, column: int):
        super().__init__(line, column)
        self.name = name
        self.declarations = declarations
        for decl in declarations:
            self.add_child(decl)


# Template Definition Node
class TemplateNode(CUDANode):
    """Template definition node"""
    def __init__(self, name: str, parameters: List[str],
                 body: CUDANode, line: int, column: int):
        super().__init__(line, column)
        self.name = name
        self.parameters = parameters
        self.body = body
        self.add_child(body)


# Root CUDA AST Node
class CudaASTNode(CUDANode):
    """Root node for CUDA AST"""
    def __init__(self):
        super().__init__(line=0, column=0)
        self.translation_unit: Optional[str] = None
        self.metal_target: Optional[str] = None
        self.optimization_level = 2


# CUDA Translation Context
class CudaTranslationContext:
    """Context for CUDA-to-Metal translation process"""
    def __init__(self, source_file: str, metal_target: str = "2.4",
                 optimization_level: int = 2):
        self.source_file = source_file
        self.metal_target = metal_target
        self.optimization_level = optimization_level
        self.type_mappings: Dict[CUDAType, str] = {}
        self.current_scope: List[str] = []
        self.buffer_index = 0
        self.used_features: Set[str] = set()
        self.thread_group_size = 256
        self.enable_fast_math = True

    def enter_scope(self, name: str) -> None:
        """Enter a new scope"""
        self.current_scope.append(name)

    def exit_scope(self) -> None:
        """Exit current scope"""
        if self.current_scope:
            self.current_scope.pop()

    def get_next_buffer_index(self) -> int:
        """Get next available buffer index"""
        index = self.buffer_index
        self.buffer_index += 1
        return index


# Convenience Type Aliases
KernelNode = CUDAKernel
ParameterNode = CUDAParameter
CompoundStmtNode = CUDACompoundStmt
