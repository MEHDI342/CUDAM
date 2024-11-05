from typing import Dict, List, Optional, Set, Any, Union, Tuple
from pathlib import Path
import threading
import asyncio
import hashlib
import re
from concurrent.futures import ThreadPoolExecutor

from ..parser.ast import (
    CudaASTNode, KernelNode, FunctionNode, VariableNode,
    CompoundStmtNode, BinaryOpNode, UnaryOpNode, CallExprNode,
    ArraySubscriptNode, IntegerLiteralNode, FloatingLiteralNode,
    DeclRefExprNode
)
from ..utils.error_handler import CudaTranslationError
from ..utils.logger import get_logger
from .memory_model_translator import MemoryModelTranslator
from .thread_hierarchy_mapper import ThreadHierarchyMapper

logger = get_logger(__name__)

@dataclass
class TranslationResult:
    metal_code: str
    entry_point: str
    buffer_bindings: Dict[str, int]
    threadgroup_size: Tuple[int, int, int]
    grid_size: Tuple[int, int, int]
    shared_memory_size: int
    metal_functions: Set[str]
    dependencies: Set[str]

class KernelTranslator:
    def __init__(self):
        self.memory_translator = MemoryModelTranslator()
        self.thread_mapper = ThreadHierarchyMapper()
        self._translation_cache: Dict[str, TranslationResult] = {}
        self._cache_lock = threading.Lock()
        self._function_registry: Dict[str, str] = {}
        self._metal_type_map = self._init_metal_type_map()
        self._barrier_points = set()

    async def translate_kernel(self, kernel: KernelNode) -> TranslationResult:
        cache_key = self._generate_cache_key(kernel)

        with self._cache_lock:
            if cache_key in self._translation_cache:
                return self._translation_cache[cache_key]

        try:
            memory_task = asyncio.create_task(
                self.memory_translator.translate_memory_model(kernel)
            )
            thread_task = asyncio.create_task(
                self.thread_mapper.map_thread_hierarchy(kernel)
            )
            signature_task = asyncio.create_task(
                self._translate_kernel_signature(kernel)
            )
            body_task = asyncio.create_task(
                self._translate_kernel_body(kernel)
            )

            memory_result, thread_result, signature, body = await asyncio.gather(
                memory_task, thread_task, signature_task, body_task
            )

            metal_code = self._generate_metal_kernel(
                kernel=kernel,
                signature=signature,
                body=body,
                memory_mappings=memory_result,
                thread_mappings=thread_result
            )

            result = TranslationResult(
                metal_code=metal_code,
                entry_point=f"metal_{kernel.name}",
                buffer_bindings=memory_result.buffer_bindings,
                threadgroup_size=thread_result.threadgroup_size,
                grid_size=thread_result.grid_size,
                shared_memory_size=memory_result.shared_memory_size,
                metal_functions=self._collect_metal_functions(body),
                dependencies=self._collect_dependencies(kernel)
            )

            with self._cache_lock:
                self._translation_cache[cache_key] = result

            return result

        except Exception as e:
            logger.error(f"Kernel translation failed: {str(e)}")
            raise CudaTranslationError(f"Failed to translate kernel {kernel.name}: {str(e)}")

    def _generate_cache_key(self, kernel: KernelNode) -> str:
        hasher = hashlib.sha256()
        self._hash_node(kernel, hasher)
        hasher.update(b"v1.0")
        hasher.update(str(self._metal_type_map).encode())
        return hasher.hexdigest()

    def _hash_node(self, node: CudaASTNode, hasher: hashlib.sha256) -> None:
        hasher.update(node.__class__.__name__.encode())
        if hasattr(node, 'name'):
            hasher.update(node.name.encode())
        if hasattr(node, 'type'):
            hasher.update(str(node.type).encode())
        for child in node.children:
            self._hash_node(child, hasher)

    async def _translate_kernel_signature(self, kernel: KernelNode) -> str:
        buffer_idx = 0
        params = []

        for param in kernel.parameters:
            metal_type = self._metal_type_map[param.type]
            if param.is_pointer():
                qualifier = "device const" if param.is_readonly() else "device"
                params.append(f"{qualifier} {metal_type}* {param.name} [[buffer({buffer_idx})]]")
            else:
                params.append(f"constant {metal_type}& {param.name} [[buffer({buffer_idx})]]")
            buffer_idx += 1

        thread_params = [
            "uint3 thread_position_in_grid [[thread_position_in_grid]]",
            "uint3 threads_per_grid [[threads_per_grid]]",
            "uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]]",
            "uint3 threadgroup_position [[threadgroup_position_in_grid]]"
        ]

        return f"kernel void metal_{kernel.name}({', '.join(params + thread_params)})"

    async def _translate_kernel_body(self, kernel: KernelNode) -> str:
        translated_stmts = []

        shared_mem = await self.memory_translator.get_shared_memory_declarations(kernel)
        if shared_mem:
            translated_stmts.extend(shared_mem)

        for stmt in kernel.body:
            metal_stmt = await self._translate_statement(stmt)
            translated_stmts.extend(metal_stmt.split('\n'))

        if self._barrier_points:
            translated_stmts.insert(0, "threadgroup_barrier(mem_flags::mem_threadgroup);")

        return '\n    '.join(translated_stmts)

    async def _translate_statement(self, stmt: CudaASTNode) -> str:
        if isinstance(stmt, CompoundStmtNode):
            return await self._translate_compound_statement(stmt)
        elif isinstance(stmt, ExpressionNode):
            return await self._translate_expression(stmt)
        elif isinstance(stmt, CallExprNode):
            return await self._translate_function_call(stmt)
        elif isinstance(stmt, ArraySubscriptNode):
            return await self._translate_array_access(stmt)
        elif isinstance(stmt, BinaryOpNode):
            return await self._translate_binary_op(stmt)
        elif isinstance(stmt, UnaryOpNode):
            return await self._translate_unary_op(stmt)
        else:
            raise CudaTranslationError(f"Unsupported statement type: {stmt.__class__.__name__}")

    async def _translate_compound_statement(self, stmt: CompoundStmtNode) -> str:
        metal_stmts = []
        for child in stmt.children:
            metal_stmt = await self._translate_statement(child)
            metal_stmts.extend(metal_stmt.split('\n'))
        return '{\n    ' + '\n    '.join(metal_stmts) + '\n}'

    async def _translate_expression(self, expr: ExpressionNode) -> str:
        if isinstance(expr, IntegerLiteralNode):
            return str(expr.value)
        elif isinstance(expr, FloatingLiteralNode):
            return str(expr.value)
        elif isinstance(expr, DeclRefExprNode):
            return expr.name
        elif isinstance(expr, BinaryOpNode):
            return await self._translate_binary_op(expr)
        elif isinstance(expr, UnaryOpNode):
            return await self._translate_unary_op(expr)
        else:
            return str(expr.value)

    async def _translate_binary_op(self, node: BinaryOpNode) -> str:
        left = await self._translate_expression(node.left)
        right = await self._translate_expression(node.right)
        return f"({left} {node.operator} {right})"

    async def _translate_unary_op(self, node: UnaryOpNode) -> str:
        operand = await self._translate_expression(node.operand)
        return f"{node.operator}({operand})"

    async def _translate_function_call(self, call: CallExprNode) -> str:
        if call.name in self._function_registry:
            metal_func = self._function_registry[call.name]
        else:
            metal_func = await self._translate_function(call)
            self._function_registry[call.name] = metal_func

        args = []
        for arg in call.arguments:
            arg_str = await self._translate_expression(arg)
            args.append(arg_str)

        return f"{metal_func}({', '.join(args)})"

    async def _translate_array_access(self, access: ArraySubscriptNode) -> str:
        array = await self._translate_expression(access.array)
        index = await self._translate_expression(access.index)

        if self.memory_translator.is_coalesced_access(access):
            return f"{array}[{index} + thread_position_in_grid.x]"
        return f"{array}[{index}]"

    def _init_metal_type_map(self) -> Dict[str, str]:
        return {
            'float': 'float',
            'double': 'float',
            'int': 'int32_t',
            'unsigned int': 'uint32_t',
            'long': 'int64_t',
            'unsigned long': 'uint64_t',
            'char': 'int8_t',
            'unsigned char': 'uint8_t',
            'short': 'int16_t',
            'unsigned short': 'uint16_t',
            'bool': 'bool',
            'void': 'void',
            'float2': 'float2',
            'float3': 'float3',
            'float4': 'float4',
            'int2': 'int2',
            'int3': 'int3',
            'int4': 'int4',
            'uint2': 'uint2',
            'uint3': 'uint3',
            'uint4': 'uint4'
        }
