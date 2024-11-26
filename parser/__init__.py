"""
CUDA Parser Module Initialization
Provides complete type system and node hierarchy for CUDA to Metal translation.

Usage:
    from CUDAM.parser import CUDAKernel, CUDAType, CUDAQualifier
"""

# Core node system imports using absolute imports
from core.parser.ast_nodes import (
    # Core node types and enums
    CUDANode,
    CUDAKernel,
    CUDAParameter,
    CUDAType,
    CUDAQualifier,
    CUDASharedMemory,
    CUDAThreadIdx,
    CUDABarrier,
    CUDACompoundStmt,
    CUDAExpressionNode,
    CUDAStatement,
    FunctionNode,
    KernelNode,
    VariableNode,
    StructNode,
    EnumNode,
    TypedefNode,
    ClassNode,
    NamespaceNode,
    TemplateNode,
    CudaASTNode,
    CudaTranslationContext
)

# Core configuration
VERSION = "1.0.0"
METAL_TARGET = "2.4"
OPTIMIZATION_LEVEL = 2

# Public API - Defines exactly what gets exported
__all__ = [
    "CUDANode",
    "CUDAKernel",
    "CUDAParameter",
    "CUDAType",
    "CUDAQualifier",
    "CUDASharedMemory",
    "CUDAThreadIdx",
    "CUDABarrier",
    "CUDACompoundStmt",
    "CUDAExpressionNode",
    "CUDAStatement",
    "FunctionNode",
    "KernelNode",
    "VariableNode",
    "StructNode",
    "EnumNode",
    "TypedefNode",
    "ClassNode",
    "NamespaceNode",
    "TemplateNode",
    "CudaASTNode",
    "CudaTranslationContext"
]

# Convenience aliases
KernelNode = CUDAKernel
ParameterNode = CUDAParameter
CompoundStmtNode = CUDACompoundStmt

# Initialize configuration
def init_translation(
        source_file: str,
        metal_target: str = METAL_TARGET,
        optimization_level: int = OPTIMIZATION_LEVEL
) -> CudaTranslationContext:
    """Initialize AST translation context with specified parameters."""
    return CudaTranslationContext(
        source_file=source_file,
        metal_target=metal_target,
        optimization_level=optimization_level
    )

# Error checking and validation
def validate_ast(node: CUDANode) -> bool:
    """Validate AST node and its children for Metal compatibility."""
    if not isinstance(node, CUDANode):
        return False
    return all(validate_ast(child) for child in node.children)
