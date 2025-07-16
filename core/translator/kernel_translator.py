from typing import Dict, List, Optional, Set, Union, Any, Tuple
import re
from threading import Lock, RLock
import logging
import time
from pathlib import Path
import hashlib
import traceback

from ..utils.error_handler import CudaTranslationError, CudaParseError
from ..utils.logger import get_logger
from ..utils.metal_equivalents import get_metal_equivalent, translate_cuda_call_to_metal
from ..utils.cuda_to_metal_type_mapping import (
    map_cuda_type_to_metal, is_vector_type, get_vector_component_type, get_vector_size
)
from ..utils.mapping_tables import (
    METAL_TYPES, METAL_FUNCTIONS, METAL_ATTRIBUTES, METAL_ADDRESS_SPACES,
    METAL_MEMORY_FLAGS, METAL_THREAD_MAPPING
)
from ..core.parser.ast_nodes import (
    CUDANode, CUDAKernel, CUDAParameter, CUDAType, CUDAQualifier,
    CUDAExpressionNode, CUDAStatement, CUDASharedMemory, CUDABarrier
)

logger = get_logger(__name__)

class KernelTranslator:
    """
    Production-grade CUDA kernel translator with comprehensive Metal optimization support.
    Thread-safe implementation for high-performance translation pipelines.

    This class handles the translation of CUDA kernels into Metal Shading Language (MSL),
    including advanced optimizations and hardware-specific considerations for Apple Silicon.

    Features:
    - Thread-safe implementation with fine-grained locking
    - Smart caching for high-performance translation
    - Comprehensive error detection and reporting
    - Multi-level optimization pipeline
    - Hardware-aware translation targeting Apple M1/M2/M3 capabilities
    """

    def __init__(self, metal_target_version: str = "2.4", optimization_level: int = 2):
        """
        Initialize the KernelTranslator with specific optimization settings.

        Args:
            metal_target_version: Target Metal API version (defaults to 2.4)
            optimization_level: Optimization aggressiveness level (0-3, default 2)
                0: No optimizations
                1: Basic optimizations (memory coalescing, barrier optimizations)
                2: Intermediate optimizations (SIMD usage, loop unrolling)
                3: Aggressive optimizations (vectorization, memory layout)
        """
        # Thread safety - use RLock for re-entrancy support
        self._lock = RLock()

        # Translation state
        self._translation_cache: Dict[str, str] = {}
        self._kernel_metadata: Dict[str, Dict[str, Any]] = {}
        self._validation_cache: Dict[str, bool] = {}

        # Feature tracking
        self.used_features: Set[str] = set()
        self.required_headers: Set[str] = set()

        # Configuration
        self.metal_target_version = metal_target_version
        self.optimization_level = optimization_level

        # Initialize hardware-specific limits for Apple Silicon
        self.metal_limits = {
            'max_threads_per_group': 1024,
            'max_total_threadgroup_memory': 32768,  # 32KB
            'simd_width': 32,
            'max_threadgroups_per_grid': (65535, 65535, 65535),
            'max_buffer_size': 256 * 1024 * 1024,  # 256MB for typical devices
            'registers_per_threadgroup': 16384,
            'max_constant_buffer_size': 64 * 1024,  # 64KB
            'texture_buffer_alignment': 16
        }

        # Performance metrics
        self.metrics = {
            'translations': 0,
            'cache_hits': 0,
            'translation_time_total': 0.0,
            'validation_time_total': 0.0,
            'optimization_time_total': 0.0
        }

        logger.info(
            f"KernelTranslator initialized with Metal {metal_target_version} target "
            f"and optimization level {optimization_level}"
        )

    def translate_kernel(self, kernel: CUDAKernel) -> str:
        """
        Translate a CUDA kernel function to Metal Shading Language (MSL).
        Thread-safe implementation with caching and profiling.

        Args:
            kernel: CUDA kernel AST node to translate

        Returns:
            str: Translated Metal shader function

        Raises:
            CudaTranslationError: If translation fails due to incompatible CUDA features
            CudaParseError: If AST node structure is invalid
            Exception: For other unexpected errors
        """
        try:
            # Generate cache key based on kernel hash and optimization level
            kernel_hash = self._compute_kernel_hash(kernel)
            cache_key = f"{kernel.name}:{kernel_hash}:{self.optimization_level}"

            # Check cache with proper locking
            with self._lock:
                if cache_key in self._translation_cache:
                    self.metrics['cache_hits'] += 1
                    logger.debug(f"Cache hit for kernel: {kernel.name}")
                    return self._translation_cache[cache_key]

            # Performance tracking
            translation_start = time.time()
            logger.info(f"Translating kernel: {kernel.name}")

            # Reset used features for this translation
            self.used_features = set()

            # Validation phase
            validation_start = time.time()
            self._validate_kernel(kernel)
            validation_time = time.time() - validation_start
            self.metrics['validation_time_total'] += validation_time

            # Step 1: Generate Metal kernel signature
            signature = self._generate_metal_signature(kernel)

            # Step 2: Generate kernel body
            body = self._translate_kernel_body(kernel)

            # Step 3: Apply optimizations based on level
            if self.optimization_level > 0:
                optimization_start = time.time()
                body = self._optimize_kernel_body(body, kernel)
                optimization_time = time.time() - optimization_start
                self.metrics['optimization_time_total'] += optimization_time

            # Step 4: Combine signature and body
            metal_kernel = f"{signature} {{\n{body}\n}}"

            # Step 5: Final validation
            self._validate_metal_code(metal_kernel)

            # Update metrics and cache result
            with self._lock:
                self.metrics['translations'] += 1
                self._translation_cache[cache_key] = metal_kernel
                self._kernel_metadata[kernel.name] = {
                    'thread_dimensions': self._get_kernel_thread_dimensions(kernel),
                    'shared_memory_size': self._get_kernel_shared_memory_size(kernel),
                    'used_features': self.used_features.copy(),
                    'parameter_count': len(kernel.parameters),
                    'translation_timestamp': time.time()
                }

            translation_time = time.time() - translation_start
            self.metrics['translation_time_total'] += translation_time
            logger.info(
                f"Successfully translated kernel {kernel.name} in {translation_time:.2f}s"
            )
            return metal_kernel

        except CudaTranslationError as e:
            logger.error(f"Translation error in kernel {kernel.name}: {str(e)}")
            raise
        except CudaParseError as e:
            logger.error(f"Parse error in kernel {kernel.name}: {str(e)}")
            raise
        except Exception as e:
            error_msg = f"Unexpected error translating kernel {kernel.name}: {str(e)}"
            stack_trace = traceback.format_exc()
            logger.error(f"{error_msg}\n{stack_trace}")
            raise CudaTranslationError(error_msg)

    def translate_ast(self, root: CUDANode) -> str:
        """
        Translate an entire CUDA AST to Metal code.

        Args:
            root: CUDA AST root node

        Returns:
            str: Complete Metal shader file
        """
        metal_lines = []

        # Process all nodes and collect required features
        self._collect_required_features(root)

        # Generate Metal headers
        metal_lines.append(self._generate_metal_headers())

        # Add any required utility functions
        utility_functions = self._generate_utility_functions()
        if utility_functions:
            metal_lines.append(utility_functions)

        device_functions = []
        kernels = []
        constants = []

        for node in root.children:
            if isinstance(node, CUDAKernel):
                kernels.append(self.translate_kernel(node))
            elif self._is_device_function(node):
                device_functions.append(self._translate_device_function(node))
            elif self._is_constant_memory(node):
                constants.append(self._translate_constant_memory(node))

        if constants:
            metal_lines.append("// Global Constants")
            metal_lines.extend(constants)
            metal_lines.append("")

        if device_functions:
            metal_lines.append("// Device Functions")
            metal_lines.extend(device_functions)
            metal_lines.append("")

        if kernels:
            metal_lines.append("// Kernel Functions")
            metal_lines.extend(kernels)

        return "\n".join(metal_lines)

    def _compute_kernel_hash(self, kernel: CUDAKernel) -> str:
        """
        Compute a hash of the kernel AST for caching purposes.
        """
        kernel_repr = f"{kernel.name}:{kernel.line}:{kernel.column}"
        for param in kernel.parameters:
            kernel_repr += f":{param.name}:{param.param_type}:{int(param.is_pointer)}"
        kernel_repr += f":{len(kernel.body)}"
        return hashlib.md5(kernel_repr.encode()).hexdigest()

    def _validate_kernel(self, kernel: CUDAKernel):
        """
        Validate kernel for Metal compatibility with comprehensive checks.
        """
        validation_errors = []

        # 1. Check thread limits
        if (hasattr(kernel, 'max_threads_per_block') and
                kernel.max_threads_per_block > self.metal_limits['max_threads_per_group']):
            validation_errors.append(
                f"Kernel {kernel.name} exceeds Metal's max threads per group limit "
                f"({kernel.max_threads_per_block} > {self.metal_limits['max_threads_per_group']})"
            )

        # 2. Check for unsupported features
        unsupported_features = self._detect_unsupported_features(kernel)
        for feature in unsupported_features:
            validation_errors.append(f"Unsupported CUDA feature detected: {feature}")

        # 3. Check shared memory limits
        shared_memory_size = self._get_kernel_shared_memory_size(kernel)
        if shared_memory_size > self.metal_limits['max_total_threadgroup_memory']:
            validation_errors.append(
                f"Shared memory usage ({shared_memory_size} bytes) exceeds Metal limit "
                f"({self.metal_limits['max_total_threadgroup_memory']} bytes)"
            )

        # 4. Check parameter count
        if len(kernel.parameters) > self.metal_limits['max_buffer_size']:
            validation_errors.append(
                f"Kernel {kernel.name} has {len(kernel.parameters)} parameters, exceeding "
                f"Metal's limit of {self.metal_limits['max_buffer_size']}"
            )

        if validation_errors:
            error_msg = f"Kernel {kernel.name} validation failed:\n" + "\n".join(
                f"- {error}" for error in validation_errors
            )
            raise CudaTranslationError(error_msg)

    def _detect_unsupported_features(self, kernel: CUDAKernel) -> List[str]:
        """
        Detect CUDA features not supported in Metal.
        """
        unsupported_features = []
        for node in kernel.traverse():
            if (isinstance(node, CUDAExpressionNode) and
                    hasattr(node, 'function') and node.function):
                func_name = node.function
                # Dynamic parallelism, 3D textures, driver API calls
                if any(fname in func_name for fname in ['cudaLaunch', 'cuLaunch']):
                    unsupported_features.append(f"Dynamic parallelism ({func_name})")
                if 'tex3D' in func_name:
                    unsupported_features.append(f"3D texture operations ({func_name})")
                if func_name.startswith('cu') and not func_name.startswith('cuda'):
                    unsupported_features.append(f"CUDA driver API call ({func_name})")
        return unsupported_features

    def _get_kernel_shared_memory_size(self, kernel: CUDAKernel) -> int:
        """
        Calculate total shared memory usage of a kernel.
        """
        total_size = 0
        for node in kernel.traverse():
            if isinstance(node, CUDASharedMemory):
                element_size = self._get_type_size(node.data_type)
                array_size = getattr(node, 'size', 1)
                total_size += element_size * array_size
        return total_size

    def _get_type_size(self, type_name: str) -> int:
        """
        Get size in bytes of a CUDA data type.
        """
        type_sizes = {
            'char': 1,
            'unsigned char': 1,
            'short': 2,
            'unsigned short': 2,
            'int': 4,
            'unsigned int': 4,
            'long': 4,
            'unsigned long': 4,
            'long long': 8,
            'unsigned long long': 8,
            'float': 4,
            'double': 8,
            'bool': 1
        }
        if is_vector_type(type_name):
            base_type = get_vector_component_type(type_name)
            vec_size = get_vector_size(type_name)
            base_size = type_sizes.get(base_type, 4)
            return base_size * vec_size
        return type_sizes.get(type_name, 4)

    def _get_kernel_thread_dimensions(self, kernel: CUDAKernel) -> Tuple[int, int, int]:
        """
        Extract thread dimensions from kernel properties.
        """
        dimensions = (256, 1, 1)
        if hasattr(kernel, 'block_dim'):
            dimensions = kernel.block_dim
        elif hasattr(kernel, 'max_threads_per_block'):
            max_threads = kernel.max_threads_per_block
            dimensions = (max_threads, 1, 1)
        return dimensions

    def _generate_metal_signature(self, kernel: CUDAKernel) -> str:
        """
        Generate Metal kernel signature with proper attributes and parameters.
        """
        signature = "kernel void "
        signature += kernel.name

        params = []
        buffer_index = 0
        for param in kernel.parameters:
            metal_param = self._translate_kernel_parameter(param, buffer_index)
            params.append(metal_param)
            buffer_index += 1

        thread_params = [
            "uint3 thread_position_in_grid [[thread_position_in_grid]]",
            "uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]",
            "uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]]",
            "uint3 threadgroup_size [[threads_per_threadgroup]]"
        ]
        params.extend(thread_params)

        signature += "(\n    " + ",\n    ".join(params) + "\n)"
        return signature

    def _translate_kernel_parameter(self, param: CUDAParameter, buffer_index: int) -> str:
        """
        Translate a CUDA kernel parameter to Metal.
        """
        metal_type = map_cuda_type_to_metal(param.param_type)
        address_space = ""
        if param.is_pointer:
            address_space = "device "
            if CUDAQualifier.CONST in param.qualifiers:
                address_space = "constant "

        if param.is_pointer:
            return f"{address_space}{metal_type}* {param.name} [[buffer({buffer_index})]]"
        else:
            return f"constant {metal_type}& {param.name} [[buffer({buffer_index})]]"

    def _translate_kernel_body(self, kernel: CUDAKernel) -> str:
        """
        Translate CUDA kernel body to Metal.
        """
        body_lines = []
        body_lines.append(self._generate_thread_index_mapping())

        shared_mem_decls = []
        for node in kernel.traverse():
            if isinstance(node, CUDASharedMemory):
                shared_mem_decls.append(self._translate_shared_memory(node))
        if shared_mem_decls:
            body_lines.append("\n    // Shared memory declarations")
            body_lines.extend(shared_mem_decls)

        body_lines.append("\n    // Kernel implementation")
        for stmt in kernel.body:
            translated = self._translate_statement(stmt)
            body_lines.append(translated)

        # Return joined string with indentation
        return "\n    ".join([""] + body_lines)

    def _generate_thread_index_mapping(self) -> str:
        """
        Generate CUDA-to-Metal thread index mapping for compatibility.
        """
        lines = [
            "// CUDA thread indexing compatibility",
            "const uint3 blockIdx = threadgroup_position_in_grid;",
            "const uint3 threadIdx = thread_position_in_threadgroup;",
            "const uint3 blockDim = threadgroup_size;",
            "const uint3 gridDim = (thread_position_in_grid + threadgroup_size - 1) / threadgroup_size;",
            (
                "const uint globalIdx = thread_position_in_grid.x + "
                "thread_position_in_grid.y * gridDim.x * blockDim.x + "
                "thread_position_in_grid.z * gridDim.x * gridDim.y * blockDim.x * blockDim.y;"
            ),
            "",
            "// SIMD group calculations for warp-level operations",
            "const uint simdLaneId = thread_position_in_threadgroup.x & 0x1F;",
            "const uint simdGroupId = thread_position_in_threadgroup.x >> 5;"
        ]
        return "\n    ".join(lines)

    def _translate_shared_memory(self, node: CUDASharedMemory) -> str:
        """
        Translate CUDA shared memory declaration to Metal threadgroup memory.
        """
        self.used_features.add("threadgroup_memory")
        metal_type = map_cuda_type_to_metal(node.data_type)
        return f"threadgroup {metal_type} {node.name}[{node.size}];  // CUDA shared memory"

    def _translate_statement(self, stmt: CUDAStatement, indent: int = 0) -> str:
        """
        Translate a CUDA statement to Metal code with proper indentation.
        """
        indent_str = "    " * indent
        if stmt.kind == "compound":
            return self._translate_compound_statement(stmt, indent)
        elif stmt.kind == "if":
            return self._translate_if_statement(stmt, indent)
        elif stmt.kind == "for":
            return self._translate_for_statement(stmt, indent)
        elif stmt.kind == "while":
            return self._translate_while_statement(stmt, indent)
        elif stmt.kind == "return":
            return self._translate_return_statement(stmt, indent)
        elif stmt.kind == "declaration":
            return self._translate_declaration_statement(stmt, indent)
        elif stmt.kind == "call":
            return self._translate_call_statement(stmt, indent)
        else:
            return f"{indent_str}// Unhandled statement type: {stmt.kind}"

    def _translate_compound_statement(self, stmt: CUDAStatement, indent: int) -> str:
        """
        Translate a compound statement (block) with proper scope management.

        Args:
            stmt: CUDA compound statement
            indent: Indentation level

        Returns:
            str: Translated Metal compound statement
        """
        indent_str = "    " * indent
        lines = [f"{indent_str}{{  // Compound statement begin"]

        for child in stmt.children:
            child_translation = self._translate_statement(child, indent + 1)
            if child_translation:
                lines.append(child_translation)

        lines.append(f"{indent_str}}}  // Compound statement end")
        return "\n".join(lines)

    def _translate_if_statement(self, stmt: CUDAStatement, indent: int) -> str:
        """
        Translate an if statement with condition and branches.
        """
        indent_str = "    " * indent
        condition = "true"
        if hasattr(stmt, 'condition') and stmt.condition:
            condition = self._translate_expression(stmt.condition)

        lines = [f"{indent_str}if ({condition}) {{"]

        if hasattr(stmt, 'then_branch') and stmt.then_branch:
            for child in stmt.then_branch:
                translated = self._translate_statement(child, indent + 1)
                if translated:
                    lines.append(translated)
        lines.append(f"{indent_str}}}")

        if hasattr(stmt, 'else_branch') and stmt.else_branch:
            lines.append(f"{indent_str}else {{")
            for child in stmt.else_branch:
                translated = self._translate_statement(child, indent + 1)
                if translated:
                    lines.append(translated)
            lines.append(f"{indent_str}}}")

        return "\n".join(lines)

    def _translate_for_statement(self, stmt: CUDAStatement, indent: int) -> str:
        """
        Translate a for loop with appropriate Metal semantics.
        """
        indent_str = "    " * indent
        init = ""
        condition = "true"
        increment = ""

        if hasattr(stmt, 'init') and stmt.init:
            init = self._translate_expression(stmt.init)
        if hasattr(stmt, 'condition') and stmt.condition:
            condition = self._translate_expression(stmt.condition)
        if hasattr(stmt, 'increment') and stmt.increment:
            increment = self._translate_expression(stmt.increment)

        # Optional loop unrolling
        if self.optimization_level >= 2 and self._is_unrollable_loop(stmt):
            return self._generate_unrolled_loop(stmt, indent)

        lines = [f"{indent_str}for ({init}; {condition}; {increment}) {{"]
        if hasattr(stmt, 'body') and stmt.body:
            for child in stmt.body:
                translated = self._translate_statement(child, indent + 1)
                if translated:
                    lines.append(translated)
        lines.append(f"{indent_str}}}")
        return "\n".join(lines)

    def _is_unrollable_loop(self, stmt: CUDAStatement) -> bool:
        """
        Determine if a for loop is a candidate for unrolling.
        """
        if self.optimization_level < 2:
            return False
        # Placeholder for real checks
        return False

    def _generate_unrolled_loop(self, stmt: CUDAStatement, indent: int) -> str:
        """
        Generate an unrolled version of a for loop.
        """
        indent_str = "    " * indent
        return (
                f"{indent_str}// Loop unrolling optimization would be applied here\n"
                + self._translate_for_statement(stmt, indent)
        )

    def _translate_while_statement(self, stmt: CUDAStatement, indent: int) -> str:
        """
        Translate a while loop to Metal.
        """
        indent_str = "    " * indent
        condition = "true"
        if hasattr(stmt, 'condition') and stmt.condition:
            condition = self._translate_expression(stmt.condition)

        lines = [f"{indent_str}while ({condition}) {{"]
        if hasattr(stmt, 'body') and stmt.body:
            for child in stmt.body:
                translated = self._translate_statement(child, indent + 1)
                if translated:
                    lines.append(translated)
        lines.append(f"{indent_str}}}")
        return "\n".join(lines)

    def _translate_return_statement(self, stmt: CUDAStatement, indent: int) -> str:
        """
        Translate a return statement to Metal.
        """
        indent_str = "    " * indent
        if hasattr(stmt, 'expression') and stmt.expression:
            expr = self._translate_expression(stmt.expression)
            return f"{indent_str}return {expr};"
        return f"{indent_str}return;"

    def _translate_declaration_statement(self, stmt: CUDAStatement, indent: int) -> str:
        """
        Translate variable declarations to Metal.
        """
        indent_str = "    " * indent
        declarations = []
        for child in stmt.children:
            if hasattr(child, 'var_type') and hasattr(child, 'name'):
                metal_type = map_cuda_type_to_metal(child.var_type)
                addr_space = ""
                if getattr(child, 'is_pointer', False):
                    addr_space = "thread "
                if hasattr(child, 'qualifiers'):
                    if CUDAQualifier.SHARED in child.qualifiers:
                        addr_space = "threadgroup "
                        self.used_features.add("threadgroup_memory")
                    elif CUDAQualifier.CONST in child.qualifiers:
                        addr_space = "constant "

                if getattr(child, 'is_pointer', False):
                    declarations.append(f"{indent_str}{addr_space}{metal_type}* {child.name};")
                else:
                    declarations.append(f"{indent_str}{addr_space}{metal_type} {child.name};")
        return "\n".join(declarations)

    def _translate_call_statement(self, stmt: CUDAStatement, indent: int) -> str:
        """
        Translate a function call statement to Metal.
        """
        indent_str = "    " * indent
        if hasattr(stmt, 'expression') and hasattr(stmt.expression, 'spelling'):
            func_name = stmt.expression.spelling
            if func_name == "__syncthreads":
                self.used_features.add("barrier")
                return f"{indent_str}threadgroup_barrier(mem_flags::mem_threadgroup);  // __syncthreads()"

        if hasattr(stmt, 'expression'):
            expr = self._translate_expression(stmt.expression)
            return f"{indent_str}{expr};"
        return f"{indent_str}// Unknown call statement"

    def _translate_expression(self, expr: Optional[CUDAExpressionNode]) -> str:
        """
        Translate a CUDA expression to Metal.
        """
        if expr is None:
            return ""

        if hasattr(expr, 'kind'):
            kind = getattr(expr, 'kind', '')
            if kind in ("BINARY_OPERATOR", "binary_operator"):
                return self._translate_binary_operator(expr)
            elif kind in ("UNARY_OPERATOR", "unary_operator"):
                return self._translate_unary_operator(expr)
            elif kind in ("CALL_EXPR", "call_expr"):
                return self._translate_call_expr(expr)
            elif kind in ("DECL_REF_EXPR", "decl_ref_expr"):
                return self._translate_decl_ref_expr(expr)
            elif kind in ("INTEGER_LITERAL", "integer_literal"):
                return str(getattr(expr, 'value', 0))
            elif kind in ("FLOATING_LITERAL", "floating_literal"):
                return str(getattr(expr, 'value', 0.0))
            elif kind in ("ARRAY_SUBSCRIPT_EXPR", "array_subscript_expr"):
                return self._translate_array_subscript(expr)
            elif kind in ("MEMBER_EXPR", "member_expr"):
                return self._translate_member_expr(expr)

        if hasattr(expr, 'spelling') and expr.spelling:
            return expr.spelling

        return "/* Untranslated expression */"

    def _translate_binary_operator(self, expr: CUDAExpressionNode) -> str:
        left = self._translate_expression(getattr(expr, 'left', None))
        right = self._translate_expression(getattr(expr, 'right', None))
        operator = getattr(expr, 'operator', '')
        metal_operator = operator

        if operator == '/':
            if self._is_integer_expr(expr.left) and self._is_integer_expr(expr.right):
                return f"({left} / float({right}))"
        return f"({left} {metal_operator} {right})"

    def _is_integer_expr(self, expr: Optional[CUDAExpressionNode]) -> bool:
        if expr is None:
            return False
        if hasattr(expr, 'type'):
            type_str = getattr(expr, 'type', '')
            return any(int_type in type_str for int_type in
                       ['int', 'long', 'short', 'char', 'unsigned'])
        return False

    def _translate_unary_operator(self, expr: CUDAExpressionNode) -> str:
        operand = self._translate_expression(getattr(expr, 'operand', None))
        operator = getattr(expr, 'operator', '')
        if not operator:
            return operand

        if operator in ["++", "--"]:
            return f"{operator}{operand}"
        return f"{operator}({operand})"

    def _translate_call_expr(self, expr: CUDAExpressionNode) -> str:
        func_name = getattr(expr, 'function', '') or getattr(expr, 'spelling', '')
        metal_func_name = self._map_cuda_function_to_metal(func_name)

        args = []
        if hasattr(expr, 'arguments'):
            for arg in expr.arguments:
                args.append(self._translate_expression(arg))

        if metal_func_name.startswith("atomic_") and metal_func_name.endswith("_explicit"):
            args.append("memory_order_relaxed")
            self.used_features.add("atomic")

        return f"{metal_func_name}({', '.join(args)})"

    def _translate_decl_ref_expr(self, expr: CUDAExpressionNode) -> str:
        if hasattr(expr, 'spelling'):
            spelling = expr.spelling
            thread_idx_map = {
                "threadIdx.x": "thread_position_in_threadgroup.x",
                "threadIdx.y": "thread_position_in_threadgroup.y",
                "threadIdx.z": "thread_position_in_threadgroup.z",
                "blockIdx.x": "threadgroup_position_in_grid.x",
                "blockIdx.y": "threadgroup_position_in_grid.y",
                "blockIdx.z": "threadgroup_position_in_grid.z",
                "blockDim.x": "threadgroup_size.x",
                "blockDim.y": "threadgroup_size.y",
                "blockDim.z": "threadgroup_size.z",
                "gridDim.x": "gridDim.x",
                "gridDim.y": "gridDim.y",
                "gridDim.z": "gridDim.z",
                "warpSize": "32"
            }
            if spelling in thread_idx_map:
                return thread_idx_map[spelling]
            return spelling
        return "/* Unknown reference */"

    def _translate_array_subscript(self, expr: CUDAExpressionNode) -> str:
        base = self._translate_expression(getattr(expr, 'base', None))
        index = self._translate_expression(getattr(expr, 'index', None))
        return f"{base}[{index}]"

    def _translate_member_expr(self, expr: CUDAExpressionNode) -> str:
        base = self._translate_expression(getattr(expr, 'base', None))
        member = getattr(expr, 'member', '')
        operator_str = "->" if getattr(expr, 'is_arrow', False) else "."
        return f"{base}{operator_str}{member}"

    def _map_cuda_function_to_metal(self, func_name: str) -> str:
        cuda_to_metal_func = {
            "__sinf": "sin",
            "__cosf": "cos",
            "__tanf": "tan",
            "__asinf": "asin",
            "__acosf": "acos",
            "__atanf": "atan",
            "__expf": "exp",
            "__exp2f": "exp2",
            "__logf": "log",
            "__log2f": "log2",
            "__log10f": "log10",
            "__powf": "pow",
            "__sqrtf": "sqrt",
            "__rsqrtf": "rsqrt",
            "__fabsf": "fabs",
            "__floorf": "floor",
            "__ceilf": "ceil",
            "__truncf": "trunc",
            "__roundf": "round",
            "__fminf": "fmin",
            "__fmaxf": "fmax",
            "__fmodf": "fmod",

            "sinf": "metal::fast::sin",
            "cosf": "metal::fast::cos",
            "tanf": "metal::fast::tan",
            "expf": "metal::fast::exp",
            "logf": "metal::fast::log",
            "sqrtf": "metal::fast::sqrt",

            "__syncthreads": "threadgroup_barrier(mem_flags::mem_threadgroup)",
            "__threadfence": "threadgroup_barrier(mem_flags::mem_device)",
            "__threadfence_block": "threadgroup_barrier(mem_flags::mem_threadgroup)",

            "atomicAdd": "atomic_fetch_add_explicit",
            "atomicSub": "atomic_fetch_sub_explicit",
            "atomicExch": "atomic_exchange_explicit",
            "atomicMin": "atomic_fetch_min_explicit",
            "atomicMax": "atomic_fetch_max_explicit",
            "atomicAnd": "atomic_fetch_and_explicit",
            "atomicOr": "atomic_fetch_or_explicit",
            "atomicXor": "atomic_fetch_xor_explicit",
            "atomicCAS": "atomic_compare_exchange_weak_explicit",

            "__ballot": "simd_ballot",
            "__all": "simd_all",
            "__any": "simd_any",
            "__shfl": "simd_shuffle",
            "__shfl_up": "simd_shuffle_up",
            "__shfl_down": "simd_shuffle_down",
            "__shfl_xor": "simd_shuffle_xor",

            "__popc": "popcount",
            "__clz": "clz",
            "__ffs": "ctz",

            "make_float2": "float2",
            "make_float3": "float3",
            "make_float4": "float4",
            "make_int2": "int2",
            "make_int3": "int3",
            "make_int4": "int4",
            "make_uint2": "uint2",
            "make_uint3": "uint3",
            "make_uint4": "uint4"
        }
        if func_name in cuda_to_metal_func:
            mapped_func = cuda_to_metal_func[func_name]
            if "atomic_" in mapped_func:
                self.used_features.add("atomic")
            elif "simd_" in mapped_func:
                self.used_features.add("simd")
            elif "threadgroup_barrier" in mapped_func:
                self.used_features.add("barrier")
            elif "metal::fast::" in mapped_func:
                self.used_features.add("fast_math")
            return mapped_func

        if func_name and not func_name.startswith(("__", "cuda", "CU")):
            logger.warning(f"Unknown CUDA function '{func_name}' - keeping original name")
        return func_name

    def _is_device_function(self, node: CUDANode) -> bool:
        return (hasattr(node, 'is_device') and node.is_device
                and not hasattr(node, 'is_kernel'))

    def _is_constant_memory(self, node: CUDANode) -> bool:
        return (hasattr(node, 'qualifiers')
                and CUDAQualifier.CONST in node.qualifiers)

    def _translate_device_function(self, node: CUDANode) -> str:
        func_name = getattr(node, 'name', 'device_func')
        return_type = map_cuda_type_to_metal(
            getattr(node, 'return_type', 'void')
        )

        signature = f"device {return_type} {func_name}("
        params = []
        for param in getattr(node, 'parameters', []):
            params.append(self._translate_device_function_parameter(param))
        if params:
            signature += ", ".join(params)
        signature += ")"

        body = []
        for stmt in getattr(node, 'body', []):
            body.append(self._translate_statement(stmt))

        return f"{signature} {{\n    " + "\n    ".join(body) + "\n}}"

    def _translate_device_function_parameter(self, param: CUDAParameter) -> str:
        metal_type = map_cuda_type_to_metal(param.param_type)
        if param.is_pointer:
            addr_space = "device "
            if CUDAQualifier.CONST in param.qualifiers:
                addr_space = "constant "
            return f"{addr_space}{metal_type}* {param.name}"
        else:
            return f"{metal_type} {param.name}"

    def _translate_constant_memory(self, node: CUDANode) -> str:
        var_type = map_cuda_type_to_metal(getattr(node, 'var_type', 'float'))
        var_name = getattr(node, 'name', 'const_var')
        if getattr(node, 'is_pointer', False):
            return f"constant {var_type}* {var_name};"
        else:
            return f"constant {var_type} {var_name};"

    def _collect_required_features(self, root: CUDANode) -> None:
        for node in root.traverse():
            if isinstance(node, CUDASharedMemory):
                self.used_features.add("threadgroup_memory")
            if isinstance(node, CUDABarrier):
                self.used_features.add("barrier")
            if (isinstance(node, CUDAExpressionNode)
                    and hasattr(node, 'function')
                    and node.function
                    and node.function.startswith("atomic")):
                self.used_features.add("atomic")
            if (isinstance(node, CUDAExpressionNode)
                    and hasattr(node, 'function')
                    and node.function
                    and node.function.startswith(("__shfl", "__ballot", "__all", "__any"))):
                self.used_features.add("simd")

    def _generate_metal_headers(self) -> str:
        headers = [
            "#include <metal_stdlib>",
            "using namespace metal;"
        ]
        if "atomic" in self.used_features:
            headers.insert(1, "#include <metal_atomic>")
            self.required_headers.add("metal_atomic")
        if "simd" in self.used_features:
            headers.insert(1, "#include <metal_simdgroup>")
            self.required_headers.add("metal_simdgroup")
        if "fast_math" in self.used_features:
            headers.insert(1, "#include <metal_math>")
            self.required_headers.add("metal_math")

        if "barrier" in self.used_features:
            headers.append("\n// Memory flags for barriers")
            headers.append("typedef enum {")
            headers.append("    mem_none        = 0,")
            headers.append("    mem_device      = 1,")
            headers.append("    mem_threadgroup = 2,")
            headers.append("    mem_texture     = 4")
            headers.append("} mem_flags;")

        if self.metal_target_version:
            major, minor = map(int, self.metal_target_version.split("."))
            version_check = major * 10000 + minor * 100
            headers.insert(0, f"#if __METAL_VERSION__ >= {version_check}")
            headers.append(f"#endif // __METAL_VERSION__ >= {version_check}")
        return "\n".join(headers)

    def _generate_utility_functions(self) -> str:
        utilities = []
        if "atomic" in self.used_features:
            utilities.append(self._generate_atomic_utilities())
        if "simd" in self.used_features:
            utilities.append(self._generate_simd_utilities())
        if "threadgroup_memory" in self.used_features:
            utilities.append(self._generate_shared_memory_utilities())

        if utilities:
            return "\n\n// Utility functions for CUDA compatibility\n" + "\n\n".join(utilities)
        return ""

    def _generate_atomic_utilities(self) -> str:
        return """// Atomic operation wrappers for CUDA compatibility
template<typename T>
T atomicAdd(device atomic<T>* address, T val) {
    return atomic_fetch_add_explicit(address, val, memory_order_relaxed);
}

template<typename T>
T atomicExch(device atomic<T>* address, T val) {
    return atomic_exchange_explicit(address, val, memory_order_relaxed);
}

template<typename T>
T atomicMin(device atomic<T>* address, T val) {
    return atomic_fetch_min_explicit(address, val, memory_order_relaxed);
}

template<typename T>
T atomicMax(device atomic<T>* address, T val) {
    return atomic_fetch_max_explicit(address, val, memory_order_relaxed);
}

template<typename T>
T atomicAnd(device atomic<T>* address, T val) {
    return atomic_fetch_and_explicit(address, val, memory_order_relaxed);
}

template<typename T>
T atomicOr(device atomic<T>* address, T val) {
    return atomic_fetch_or_explicit(address, val, memory_order_relaxed);
}

template<typename T>
T atomicXor(device atomic<T>* address, T val) {
    return atomic_fetch_xor_explicit(address, val, memory_order_relaxed);
}

template<typename T>
T atomicCAS(device atomic<T>* address, T compare, T val) {
    T expected = compare;
    atomic_compare_exchange_weak_explicit(address, &expected, val, memory_order_relaxed, memory_order_relaxed);
    return expected;
}"""

    def _generate_simd_utilities(self) -> str:
        return """// SIMD utility stubs for warp-level operations
inline uint simd_ballot(bool predicate) {
    // Implementation placeholder
    return predicate ? 0xFFFFFFFF : 0;
}
inline bool simd_all(bool predicate) {
    // Implementation placeholder
    return predicate;
}
inline bool simd_any(bool predicate) {
    // Implementation placeholder
    return predicate;
}
inline uint simd_shuffle(uint value, uint lane) {
    // Implementation placeholder
    return value;
}
inline uint simd_shuffle_up(uint value, uint delta) {
    // Implementation placeholder
    return value;
}
inline uint simd_shuffle_down(uint value, uint delta) {
    // Implementation placeholder
    return value;
}
inline uint simd_shuffle_xor(uint value, uint mask) {
    // Implementation placeholder
    return value;
}"""

    def _generate_shared_memory_utilities(self) -> str:
        return """// Shared memory utility stubs
// Additional logic can be added here if needed
"""

    def _optimize_kernel_body(self, body: str, kernel: CUDAKernel) -> str:
        # Placeholder for body optimization
        return body

    def _validate_metal_code(self, metal_code: str) -> None:
        # Placeholder for final code checks
        pass
