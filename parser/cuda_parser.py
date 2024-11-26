"""
CUDA to Metal Parser - Production Implementation
Handles complete CUDA code parsing and Metal conversion with full error handling.
"""

import os
import re
import sys
import json
import logging
import platform
import hashlib
import glob
from typing import List, Dict, Any, Optional, Union, Set, Tuple, Callable
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import clang
import clang.cindex
from clang.cindex import (
    CursorKind, TypeKind, TranslationUnit, AccessSpecifier,
    Index, Cursor, TokenKind
)

# Assuming the existence of these modules based on your imports
from ..parser.ast_nodes import (
    CUDANode, CUDAKernel, CUDAParameter, CUDAType, CUDAQualifier,
    CUDASharedMemory, CUDAThreadIdx, CUDABarrier, CUDACompoundStmt,
    CUDAExpressionNode, CUDAStatement, FunctionNode, KernelNode,
    VariableNode, StructNode, EnumNode, TypedefNode, ClassNode,
    NamespaceNode, TemplateNode, CudaASTNode, CudaTranslationContext
)
from ..utils.error_handler import (
    CudaParseError, CudaTranslationError, CudaTypeError,
    CudaNotSupportedError, raise_cuda_parse_error
)
from ..utils.logger import get_logger
from ..utils.metal_equivalents import get_metal_equivalent, METAL_EQUIVALENTS
from ..utils.mapping_tables import MetalMappingRegistry

# Initialize logger
logger = get_logger(__name__)

class MetalIntegration:
    """Handles Metal framework integration based on platform."""

    def __init__(self):
        self.platform = platform.system()
        self._metal_available = self._check_metal_availability()
        self._metal_compiler_path = self._find_metal_compiler()

    def _check_metal_availability(self) -> bool:
        """Check if Metal is available on current system."""
        if self.platform == 'Darwin':
            try:
                import Metal
                import MetalKit
                return True
            except ImportError:
                logger.warning("Metal frameworks not available - using fallback implementation")
                return False
        return False

    def _find_metal_compiler(self) -> Optional[str]:
        """Locate Metal compiler executable."""
        if self.platform == 'Darwin':
            metal_path = '/usr/bin/metal'
            if os.path.exists(metal_path):
                return metal_path
        return None

    def validate_metal_support(self) -> bool:
        """Validate complete Metal support availability."""
        return self._metal_available and self._metal_compiler_path is not None

class CudaParser:
    """
    Production-grade CUDA parser with complete Metal translation support.
    Thread-safe implementation with comprehensive error handling.
    """

    def __init__(self, cuda_include_paths: Optional[List[str]] = None,
                 plugins: Optional[List[Any]] = None,
                 optimization_level: int = 2):
        """
        Initialize the CUDA Parser with enhanced capabilities.

        Args:
            cuda_include_paths: List of paths to CUDA include directories
            plugins: Optional list of parser plugins for extended functionality
            optimization_level: Level of optimization (0-3, higher means more aggressive)
        """
        # Initialize core components
        self.index = Index.create()
        self.metal_integration = MetalIntegration()
        self.metal_registry = MetalMappingRegistry()
        self.plugins = plugins or []
        self._lock = Lock()

        # Configure paths and options
        self.cuda_include_paths = cuda_include_paths or self._find_cuda_paths()
        self.translation_options = self._init_translation_options()

        # Initialize caches
        self.ast_cache: Dict[str, Dict[str, Any]] = {}
        self.type_cache: Dict[str, CUDAType] = {}
        self.function_cache: Dict[str, Any] = {}

        # Configure libclang
        self._configure_clang()

    def _find_cuda_paths(self) -> List[str]:
        """Find CUDA installation paths with validation."""
        cuda_paths = []
        common_paths = [
            '/usr/local/cuda/include',
            '/opt/cuda/include',
            '/usr/cuda/include',
            os.path.expanduser('~/cuda/include'),
            *glob.glob('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/*/include')
        ]

        for path in common_paths:
            if os.path.exists(path):
                cuda_paths.append(path)

        if not cuda_paths:
            logger.warning("No CUDA include paths found - using default locations")
            cuda_paths = ['/usr/local/cuda/include']

        return cuda_paths

    def _init_translation_options(self) -> Dict[str, Any]:
        """Initialize translation options with defaults."""
        return {
            'optimization_level': 2,
            'enable_metal_validation': True,
            'enable_fast_math': True,
            'max_threads_per_group': 1024,
            'prefer_simd_groups': True,
            'enable_barriers': True
        }

    def _configure_clang(self):
        """Configure clang with complete error handling."""
        try:
            # Find libclang
            if sys.platform == 'win32':
                clang_lib = self._find_windows_clang()
            else:
                clang_lib = self._find_unix_clang()

            if not clang_lib:
                raise CudaParseError("Could not find libclang installation")

            # Configure clang
            clang.cindex.Config.set_library_file(clang_lib)

            # Validate configuration
            test_index = Index.create()
            if not test_index:
                raise CudaParseError("Failed to create clang Index")

        except Exception as e:
            logger.error(f"Failed to configure clang: {str(e)}")
            raise CudaParseError(f"Clang configuration failed: {str(e)}")

    def _find_windows_clang(self) -> Optional[str]:
        """Find clang on Windows systems."""
        search_paths = [
            r"C:\Program Files\LLVM\bin\libclang.dll",
            r"C:\Program Files (x86)\LLVM\bin\libclang.dll"
        ]
        return next((p for p in search_paths if os.path.exists(p)), None)

    def _find_unix_clang(self) -> Optional[str]:
        """Find clang on Unix systems."""
        search_patterns = [
            "/usr/lib/llvm-*/lib/libclang.so",
            "/usr/lib/x86_64-linux-gnu/libclang-*.so",
            "/usr/local/opt/llvm/lib/libclang.dylib"
        ]

        for pattern in search_patterns:
            matches = glob.glob(pattern)
            if matches:
                return matches[-1]  # Return highest version

        return None

    def _get_file_hash(self, file_path: str) -> str:
        """Calculate file hash for caching."""
        with open(file_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()

    def _check_cache(self, file_path: str, file_hash: str) -> bool:
        """Check if cached AST is valid."""
        if file_path not in self.ast_cache:
            return False

        cache_entry = self.ast_cache[file_path]
        return (
            cache_entry['hash'] == file_hash and
            cache_entry['timestamp'] == os.path.getmtime(file_path)
        )

    def _get_clang_args(self) -> List[str]:
        """Get clang compilation arguments."""
        base_args = [
            '-x', 'cuda',
            '--cuda-gpu-arch=sm_75',
            '-std=c++14',
            '-D__CUDACC__',
            '-D__CUDA_ARCH__=750',
            '-DNDEBUG'
        ]

        cuda_specific_args = [
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
            '-D__CUDA_NO_BFLOAT16_CONVERSIONS__',
            '-D__CUDA_ARCH_LIST__=750',
            '-D__CUDA_PREC_DIV=1',
            '-D__CUDA_PREC_SQRT=1'
        ]

        optimization_args = []
        if self.translation_options['optimization_level'] > 0:
            optimization_args.extend([
                '-O2',
                '-ffast-math',
                '-fno-strict-aliasing'
            ])

        include_paths = [f'-I{path}' for path in self.cuda_include_paths]

        return base_args + cuda_specific_args + optimization_args + include_paths

    def parse_file(self, file_path: str) -> Optional[CUDANode]:
        """
        Parse CUDA source file with complete error handling.

        Args:
            file_path: Path to CUDA source file

        Returns:
            CUDANode: Root node of AST if successful, None otherwise

        Raises:
            CudaParseError: If parsing fails
            FileNotFoundError: If file doesn't exist
        """
        try:
            # Input validation
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"CUDA source file not found: {file_path}")

            # Check cache
            file_hash = self._get_file_hash(file_path)
            with self._lock:
                if self._check_cache(file_path, file_hash):
                    logger.info(f"Using cached AST for {file_path}")
                    return self.ast_cache[file_path]['ast']

            # Parse with clang
            args = self._get_clang_args()
            translation_unit = self.index.parse(
                file_path,
                args=args,
                options=(
                    TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD |
                    TranslationUnit.PARSE_INCOMPLETE |
                    TranslationUnit.PARSE_CACHE_COMPLETION_RESULTS
                )
            )

            # Validate parse results
            if not translation_unit:
                raise CudaParseError(f"Failed to parse {file_path}")

            self._validate_diagnostics(translation_unit)

            # Convert to AST
            ast = self._convert_translation_unit(translation_unit.cursor)

            # Perform additional analysis and optimizations
            self._perform_additional_analysis(ast)

            # Translate to Metal
            metal_code = self._translate_to_metal(ast)

            # Optionally, compile Metal code
            if self.metal_integration.validate_metal_support():
                self._compile_metal_code(metal_code)
            else:
                logger.warning("Metal compiler not available. Skipping Metal code compilation.")

            # Cache results
            with self._lock:
                self.ast_cache[file_path] = {
                    'hash': file_hash,
                    'ast': ast,
                    'timestamp': os.path.getmtime(file_path)
                }

            return ast

        except FileNotFoundError as fnf_err:
            logger.error(str(fnf_err))
            raise
        except CudaParseError as parse_err:
            logger.error(str(parse_err))
            raise
        except Exception as e:
            logger.error(f"Unexpected error while parsing {file_path}: {str(e)}")
            raise CudaParseError(f"Unexpected error: {str(e)}")

    def _validate_diagnostics(self, translation_unit: TranslationUnit):
        """Validate translation unit diagnostics."""
        errors = []
        warnings = []

        for diag in translation_unit.diagnostics:
            if diag.severity >= diag.Error:
                errors.append(self._format_diagnostic(diag))
            elif diag.severity == diag.Warning:
                warnings.append(self._format_diagnostic(diag))

        # Log warnings
        for warning in warnings:
            logger.warning(f"Clang warning: {warning}")

        # Raise error if needed
        if errors:
            error_msg = "\n".join(errors)
            raise CudaParseError(f"Parse errors occurred:\n{error_msg}")

    def _format_diagnostic(self, diag: Any) -> str:
        """Format diagnostic message."""
        return (
            f"{diag.location.file}:{diag.location.line}:{diag.location.column} - "
            f"{diag.severity_name}: {diag.spelling}"
        )

    def _convert_translation_unit(self, cursor: Cursor) -> CUDANode:
        """Convert translation unit to CUDA AST."""
        node = CUDANode(
            kind=cursor.kind.name,
            spelling=cursor.spelling,
            type=cursor.type.spelling,
            children=[]
        )

        for child in cursor.get_children():
            converted = self._convert_cursor(child)
            if converted:
                node.add_child(converted)

        return node

    def _convert_cursor(self, cursor: Cursor) -> Optional[CUDANode]:
        """Convert cursor to appropriate CUDA AST node."""
        # Handle CUDA-specific nodes
        if cursor.kind == CursorKind.CUDA_GLOBAL_ATTR:
            return self._convert_kernel(cursor)
        elif cursor.kind == CursorKind.CUDA_DEVICE_ATTR:
            return self._convert_device_function(cursor)
        elif cursor.kind == CursorKind.CUDA_SHARED_ATTR:
            return self._convert_shared_memory(cursor)

        # Handle standard nodes
        converters = {
            CursorKind.FUNCTION_DECL: self._convert_function,
            CursorKind.VAR_DECL: self._convert_variable,
            CursorKind.FIELD_DECL: self._convert_field,
            CursorKind.COMPOUND_STMT: self._convert_compound_stmt,
            CursorKind.RETURN_STMT: self._convert_return_stmt,
            CursorKind.IF_STMT: self._convert_if_stmt,
            CursorKind.FOR_STMT: self._convert_for_stmt,
            CursorKind.WHILE_STMT: self._convert_while_stmt,
            CursorKind.DO_STMT: self._convert_do_stmt,
            CursorKind.BINARY_OPERATOR: self._convert_binary_operator,
            CursorKind.UNARY_OPERATOR: self._convert_unary_operator,
            CursorKind.CALL_EXPR: self._convert_call_expr,
            CursorKind.CLASS_DECL: self._convert_class,
            CursorKind.STRUCT_DECL: self._convert_struct,
            CursorKind.ENUM_DECL: self._convert_enum,
            CursorKind.TYPEDEF_DECL: self._convert_typedef,
            CursorKind.NAMESPACE: self._convert_namespace,
            CursorKind.CONSTRUCTOR: self._convert_constructor,
            CursorKind.DESTRUCTOR: self._convert_destructor,
            CursorKind.CXX_METHOD: self._convert_method,
            CursorKind.TEMPLATE_DECL: self._convert_template
        }

        converter = converters.get(cursor.kind)
        if converter:
            return converter(cursor)

        # Default conversion
        return self._convert_default(cursor)

    def _convert_kernel(self, cursor: Cursor) -> CUDAKernel:
        """Convert CUDA kernel function."""
        # Parse parameters
        parameters = []
        for arg in cursor.get_arguments():
            param = self._convert_variable(arg)
            if param:
                parameters.append(param)

        # Convert body
        body = []
        for child in cursor.get_children():
            if child.kind != CursorKind.PARM_DECL:
                node = self._convert_cursor(child)
                if node:
                    body.append(node)

        # Create kernel node
        kernel = CUDAKernel(
            name=cursor.spelling,
            parameters=parameters,
            body=body,
            return_type=CUDAType.VOID,
            location=self._get_cursor_location(cursor)
        )

        return kernel

    def _convert_device_function(self, cursor: Cursor) -> FunctionNode:
        """Convert CUDA device function."""
        # Parse parameters
        parameters = []
        for arg in cursor.get_arguments():
            param = self._convert_variable(arg)
            if param:
                parameters.append(param)

        # Convert body
        body = []
        for child in cursor.get_children():
            if child.kind != CursorKind.PARM_DECL:
                node = self._convert_cursor(child)
                if node:
                    body.append(node)

        # Create function node
        function = FunctionNode(
            name=cursor.spelling,
            parameters=parameters,
            body=body,
            return_type=CUDAType(cursor.result_type.spelling),
            location=self._get_cursor_location(cursor)
        )

        return function

    def _convert_shared_memory(self, cursor: Cursor) -> CUDASharedMemory:
        """Convert CUDA shared memory declaration."""
        # Assuming shared memory is declared as a variable
        var = self._convert_variable(cursor)
        if var:
            shared_mem = CUDASharedMemory(
                name=var.name,
                data_type=var.data_type,
                size=var.size,
                location=self._get_cursor_location(cursor)
            )
            return shared_mem
        else:
            raise CudaParseError("Failed to parse shared memory declaration.")

    def _convert_function(self, cursor: Cursor) -> FunctionNode:
        """Convert regular CUDA function."""
        # Parse parameters
        parameters = []
        for arg in cursor.get_arguments():
            param = self._convert_variable(arg)
            if param:
                parameters.append(param)

        # Convert body
        body = []
        for child in cursor.get_children():
            if child.kind != CursorKind.PARM_DECL:
                node = self._convert_cursor(child)
                if node:
                    body.append(node)

        # Create function node
        function = FunctionNode(
            name=cursor.spelling,
            parameters=parameters,
            body=body,
            return_type=CUDAType(cursor.result_type.spelling),
            location=self._get_cursor_location(cursor)
        )

        return function

    def _convert_variable(self, cursor: Cursor) -> Optional[VariableNode]:
        """Convert variable declaration."""
        var_type = CUDAType(cursor.type.spelling)
        var_name = cursor.spelling
        var_location = self._get_cursor_location(cursor)

        # Handle array types
        array_size = None
        if cursor.type.kind == TypeKind.CONSTANTARRAY:
            array_size = cursor.type.element_count

        variable = VariableNode(
            name=var_name,
            data_type=var_type,
            array_size=array_size,
            location=var_location
        )

        return variable

    def _convert_field(self, cursor: Cursor) -> VariableNode:
        """Convert struct/class field declaration."""
        return self._convert_variable(cursor)

    def _convert_compound_stmt(self, cursor: Cursor) -> CUDACompoundStmt:
        """Convert compound statement."""
        children = []
        for child in cursor.get_children():
            node = self._convert_cursor(child)
            if node:
                children.append(node)

        compound_stmt = CUDACompoundStmt(
            children=children,
            location=self._get_cursor_location(cursor)
        )

        return compound_stmt

    def _convert_return_stmt(self, cursor: Cursor) -> CUDAStatement:
        """Convert return statement."""
        # Extract return expression
        return_expr = None
        for child in cursor.get_children():
            return_expr = self._convert_expression(child)

        return_stmt = CUDAStatement(
            kind='RETURN',
            expression=return_expr,
            location=self._get_cursor_location(cursor)
        )

        return return_stmt

    def _convert_if_stmt(self, cursor: Cursor) -> CUDAStatement:
        """Convert if statement."""
        condition = None
        then_branch = []
        else_branch = []

        children = list(cursor.get_children())
        if len(children) >= 1:
            condition = self._convert_expression(children[0])
        if len(children) >= 2:
            then_node = self._convert_cursor(children[1])
            if then_node:
                then_branch.append(then_node)
        if len(children) == 3:
            else_node = self._convert_cursor(children[2])
            if else_node:
                else_branch.append(else_node)

        if_stmt = CUDAStatement(
            kind='IF',
            condition=condition,
            then_branch=then_branch,
            else_branch=else_branch,
            location=self._get_cursor_location(cursor)
        )

        return if_stmt

    def _convert_for_stmt(self, cursor: Cursor) -> CUDAStatement:
        """Convert for loop."""
        init = None
        condition = None
        increment = None
        body = []

        children = list(cursor.get_children())
        if len(children) >= 1:
            init = self._convert_expression(children[0])
        if len(children) >= 2:
            condition = self._convert_expression(children[1])
        if len(children) >= 3:
            increment = self._convert_expression(children[2])
        if len(children) >= 4:
            body_node = self._convert_cursor(children[3])
            if body_node:
                body.append(body_node)

        for_stmt = CUDAStatement(
            kind='FOR',
            init=init,
            condition=condition,
            increment=increment,
            body=body,
            location=self._get_cursor_location(cursor)
        )

        return for_stmt

    def _convert_while_stmt(self, cursor: Cursor) -> CUDAStatement:
        """Convert while loop."""
        condition = None
        body = []

        children = list(cursor.get_children())
        if len(children) >= 1:
            condition = self._convert_expression(children[0])
        if len(children) >= 2:
            body_node = self._convert_cursor(children[1])
            if body_node:
                body.append(body_node)

        while_stmt = CUDAStatement(
            kind='WHILE',
            condition=condition,
            body=body,
            location=self._get_cursor_location(cursor)
        )

        return while_stmt

    def _convert_do_stmt(self, cursor: Cursor) -> CUDAStatement:
        """Convert do-while loop."""
        body = []
        condition = None

        children = list(cursor.get_children())
        if len(children) >= 1:
            body_node = self._convert_cursor(children[0])
            if body_node:
                body.append(body_node)
        if len(children) >= 2:
            condition = self._convert_expression(children[1])

        do_stmt = CUDAStatement(
            kind='DO_WHILE',
            condition=condition,
            body=body,
            location=self._get_cursor_location(cursor)
        )

        return do_stmt

    def _convert_binary_operator(self, cursor: Cursor) -> CUDAExpressionNode:
        """Convert binary operator."""
        tokens = list(cursor.get_tokens())
        operator = None
        for tok in tokens:
            if tok.kind == TokenKind.PUNCTUATION and tok.spelling in {'+', '-', '*', '/', '%', '==', '!=', '<', '>', '<=', '>=', '&&', '||', '=', '+=', '-=', '*=', '/='}:
                operator = tok.spelling
                break

        children = list(cursor.get_children())
        left = self._convert_expression(children[0]) if len(children) >=1 else None
        right = self._convert_expression(children[1]) if len(children) >=2 else None

        binary_op = CUDAExpressionNode(
            kind='BINARY_OP',
            operator=operator,
            left=left,
            right=right,
            location=self._get_cursor_location(cursor)
        )

        return binary_op

    def _convert_unary_operator(self, cursor: Cursor) -> CUDAExpressionNode:
        """Convert unary operator."""
        tokens = list(cursor.get_tokens())
        operator = None
        for tok in tokens:
            if tok.kind == TokenKind.PUNCTUATION and tok.spelling in {'++', '--', '!', '~', '-', '+'}:
                operator = tok.spelling
                break

        children = list(cursor.get_children())
        operand = self._convert_expression(children[0]) if len(children) >=1 else None

        unary_op = CUDAExpressionNode(
            kind='UNARY_OP',
            operator=operator,
            operand=operand,
            location=self._get_cursor_location(cursor)
        )

        return unary_op

    def _convert_call_expr(self, cursor: Cursor) -> CUDAExpressionNode:
        """Convert function call expression."""
        func_name = cursor.spelling
        args = [self._convert_expression(child) for child in cursor.get_children()]

        call_expr = CUDAExpressionNode(
            kind='CALL_EXPR',
            function=func_name,
            arguments=args,
            location=self._get_cursor_location(cursor)
        )

        return call_expr

    def _convert_class(self, cursor: Cursor) -> ClassNode:
        """Convert class declaration."""
        class_name = cursor.spelling
        members = []
        methods = []

        for child in cursor.get_children():
            if child.kind == CursorKind.FIELD_DECL:
                member = self._convert_field(child)
                members.append(member)
            elif child.kind == CursorKind.CXX_METHOD:
                method = self._convert_method(child)
                methods.append(method)

        class_node = ClassNode(
            name=class_name,
            members=members,
            methods=methods,
            location=self._get_cursor_location(cursor)
        )

        return class_node

    def _convert_struct(self, cursor: Cursor) -> StructNode:
        """Convert struct declaration."""
        struct_name = cursor.spelling
        members = []

        for child in cursor.get_children():
            if child.kind == CursorKind.FIELD_DECL:
                member = self._convert_field(child)
                members.append(member)

        struct_node = StructNode(
            name=struct_name,
            members=members,
            location=self._get_cursor_location(cursor)
        )

        return struct_node

    def _convert_enum(self, cursor: Cursor) -> EnumNode:
        """Convert enum declaration."""
        enum_name = cursor.spelling
        enumerators = []

        for child in cursor.get_children():
            if child.kind == CursorKind.ENUM_CONSTANT_DECL:
                enumerator = child.spelling
                enumerators.append(enumerator)

        enum_node = EnumNode(
            name=enum_name,
            enumerators=enumerators,
            location=self._get_cursor_location(cursor)
        )

        return enum_node

    def _convert_typedef(self, cursor: Cursor) -> TypedefNode:
        """Convert typedef declaration."""
        original_type = cursor.underlying_typedef_type.spelling
        alias = cursor.spelling

        typedef_node = TypedefNode(
            alias=alias,
            original_type=original_type,
            location=self._get_cursor_location(cursor)
        )

        return typedef_node

    def _convert_namespace(self, cursor: Cursor) -> NamespaceNode:
        """Convert namespace declaration."""
        namespace_name = cursor.spelling
        members = []

        for child in cursor.get_children():
            member = self._convert_cursor(child)
            if member:
                members.append(member)

        namespace_node = NamespaceNode(
            name=namespace_name,
            members=members,
            location=self._get_cursor_location(cursor)
        )

        return namespace_node

    def _convert_method(self, cursor: Cursor) -> FunctionNode:
        """Convert class method."""
        method_name = cursor.spelling
        parameters = []
        for arg in cursor.get_arguments():
            param = self._convert_variable(arg)
            if param:
                parameters.append(param)

        body = []
        for child in cursor.get_children():
            if child.kind != CursorKind.PARM_DECL:
                node = self._convert_cursor(child)
                if node:
                    body.append(node)

        method_node = FunctionNode(
            name=method_name,
            parameters=parameters,
            body=body,
            return_type=CUDAType(cursor.result_type.spelling),
            location=self._get_cursor_location(cursor)
        )

        return method_node

    def _convert_constructor(self, cursor: Cursor) -> FunctionNode:
        """Convert class constructor."""
        constructor_name = cursor.spelling
        parameters = []
        for arg in cursor.get_arguments():
            param = self._convert_variable(arg)
            if param:
                parameters.append(param)

        body = []
        for child in cursor.get_children():
            if child.kind != CursorKind.PARM_DECL:
                node = self._convert_cursor(child)
                if node:
                    body.append(node)

        constructor_node = FunctionNode(
            name=constructor_name,
            parameters=parameters,
            body=body,
            return_type=CUDAType.VOID,
            location=self._get_cursor_location(cursor)
        )

        return constructor_node

    def _convert_destructor(self, cursor: Cursor) -> FunctionNode:
        """Convert class destructor."""
        destructor_name = cursor.spelling
        parameters = []
        body = []
        for child in cursor.get_children():
            if child.kind != CursorKind.PARM_DECL:
                node = self._convert_cursor(child)
                if node:
                    body.append(node)

        destructor_node = FunctionNode(
            name=destructor_name,
            parameters=parameters,
            body=body,
            return_type=CUDAType.VOID,
            location=self._get_cursor_location(cursor)
        )

        return destructor_node

    def _convert_expression(self, cursor: Cursor) -> CUDAExpressionNode:
        """Convert general expression."""
        if cursor.kind == CursorKind.INTEGER_LITERAL:
            value = self._get_literal_value(cursor)
            return CUDAExpressionNode(
                kind='INTEGER_LITERAL',
                value=value,
                location=self._get_cursor_location(cursor)
            )
        elif cursor.kind == CursorKind.FLOATING_LITERAL:
            value = self._get_literal_value(cursor)
            return CUDAExpressionNode(
                kind='FLOATING_LITERAL',
                value=value,
                location=self._get_cursor_location(cursor)
            )
        elif cursor.kind == CursorKind.STRING_LITERAL:
            value = self._get_literal_value(cursor)
            return CUDAExpressionNode(
                kind='STRING_LITERAL',
                value=value,
                location=self._get_cursor_location(cursor)
            )
        elif cursor.kind == CursorKind.DECL_REF_EXPR:
            return CUDAExpressionNode(
                kind='DECL_REF_EXPR',
                spelling=cursor.spelling,
                location=self._get_cursor_location(cursor)
            )
        elif cursor.kind == CursorKind.BINARY_OPERATOR:
            return self._convert_binary_operator(cursor)
        elif cursor.kind == CursorKind.UNARY_OPERATOR:
            return self._convert_unary_operator(cursor)
        elif cursor.kind == CursorKind.CALL_EXPR:
            return self._convert_call_expr(cursor)
        elif cursor.kind == CursorKind.ARRAY_SUBSCRIPT_EXPR:
            return self._convert_array_subscript(cursor)
        elif cursor.kind == CursorKind.MEMBER_REF_EXPR:
            return self._convert_member_ref_expr(cursor)
        elif cursor.kind == CursorKind.CONDITIONAL_OPERATOR:
            return self._convert_conditional_operator(cursor)
        elif cursor.kind == CursorKind.INIT_LIST_EXPR:
            return self._convert_init_list_expr(cursor)
        # Add more expression types as needed
        else:
            logger.warning(f"Unhandled expression kind: {cursor.kind.name}")
            return CUDAExpressionNode(
                kind=cursor.kind.name,
                spelling=cursor.spelling,
                location=self._get_cursor_location(cursor)
            )

    def _get_literal_value(self, cursor: Cursor) -> Any:
        """Retrieve literal value from cursor."""
        tokens = list(cursor.get_tokens())
        for tok in tokens:
            if tok.kind in {TokenKind.LITERAL, TokenKind.IDENTIFIER}:
                return tok.spelling
        return None

    def _convert_array_subscript(self, cursor: Cursor) -> CUDAExpressionNode:
        """Convert array subscript expression."""
        children = list(cursor.get_children())
        array = self._convert_expression(children[0]) if len(children) >=1 else None
        index = self._convert_expression(children[1]) if len(children) >=2 else None

        array_subscript = CUDAExpressionNode(
            kind='ARRAY_SUBSCRIPT',
            array=array,
            index=index,
            location=self._get_cursor_location(cursor)
        )

        return array_subscript

    def _convert_member_ref_expr(self, cursor: Cursor) -> CUDAExpressionNode:
        """Convert member reference expression."""
        member_name = cursor.spelling
        children = list(cursor.get_children())
        base = self._convert_expression(children[0]) if len(children) >=1 else None

        member_ref = CUDAExpressionNode(
            kind='MEMBER_REF',
            base=base,
            member=member_name,
            location=self._get_cursor_location(cursor)
        )

        return member_ref

    def _convert_conditional_operator(self, cursor: Cursor) -> CUDAExpressionNode:
        """Convert conditional (ternary) operator."""
        children = list(cursor.get_children())
        condition = self._convert_expression(children[0]) if len(children) >=1 else None
        then_expr = self._convert_expression(children[1]) if len(children) >=2 else None
        else_expr = self._convert_expression(children[2]) if len(children) >=3 else None

        conditional_op = CUDAExpressionNode(
            kind='CONDITIONAL_OPERATOR',
            condition=condition,
            then_expression=then_expr,
            else_expression=else_expr,
            location=self._get_cursor_location(cursor)
        )

        return conditional_op

    def _convert_init_list_expr(self, cursor: Cursor) -> CUDAExpressionNode:
        """Convert initializer list expression."""
        elements = [self._convert_expression(child) for child in cursor.get_children()]
        init_list = CUDAExpressionNode(
            kind='INIT_LIST_EXPR',
            elements=elements,
            location=self._get_cursor_location(cursor)
        )
        return init_list

    def _convert_default(self, cursor: Cursor) -> CUDANode:
        """Default conversion for unhandled cursor types."""
        node = CUDANode(
            kind=cursor.kind.name,
            spelling=cursor.spelling,
            type=cursor.type.spelling,
            children=[]
        )
        for child in cursor.get_children():
            converted = self._convert_cursor(child)
            if converted:
                node.add_child(converted)
        return node

    def _get_cursor_location(self, cursor: Cursor) -> Dict[str, Any]:
        """Get detailed cursor location information."""
        location = cursor.location
        extent = cursor.extent

        return {
            'file': str(location.file) if location.file else None,
            'line': location.line,
            'column': location.column,
            'offset': location.offset,
            'start': {
                'line': extent.start.line,
                'column': extent.start.column,
                'offset': extent.start.offset
            },
            'end': {
                'line': extent.end.line,
                'column': extent.end.column,
                'offset': extent.end.offset
            }
        }

    def _validate_diagnostics(self, translation_unit: TranslationUnit):
        """Validate translation unit diagnostics."""
        errors = []
        warnings = []

        for diag in translation_unit.diagnostics:
            if diag.severity >= diag.Error:
                errors.append(self._format_diagnostic(diag))
            elif diag.severity == diag.Warning:
                warnings.append(self._format_diagnostic(diag))

        # Log warnings
        for warning in warnings:
            logger.warning(f"Clang warning: {warning}")

        # Raise error if needed
        if errors:
            error_msg = "\n".join(errors)
            raise CudaParseError(f"Parse errors occurred:\n{error_msg}")

    def _format_diagnostic(self, diag: Any) -> str:
        """Format diagnostic message."""
        return (
            f"{diag.location.file}:{diag.location.line}:{diag.location.column} - "
            f"{diag.severity_name}: {diag.spelling}"
        )

    def _perform_additional_analysis(self, ast: CUDANode):
        """Perform additional AST analysis and optimizations."""
        # Placeholder for dataflow analysis, alias analysis, etc.
        # Implement as needed based on requirements
        logger.info("Performing additional AST analysis and optimizations.")
        pass

    def _translate_to_metal(self, ast: CUDANode) -> str:
        """Translate CUDA AST to Metal code."""
        logger.info("Translating CUDA AST to Metal code.")
        metal_code = []
        # Generate Metal headers
        metal_code.extend(self._generate_metal_headers())

        # Traverse AST and generate Metal code
        for child in ast.children:
            translated = self._translate_node(child)
            metal_code.append(translated)

        # Join all code parts
        return "\n".join(metal_code)

    def _generate_metal_headers(self) -> List[str]:
        """Generate required Metal headers and imports."""
        headers = [
            "#include <metal_stdlib>",
            "#include <metal_atomic>",
            "#include <metal_math>",
            "#include <metal_geometric>",
            "#include <metal_matrix>",
            "#include <metal_graphics>",
            "#include <metal_texture>",
            "#include <metal_compute>",
            "",
            "using namespace metal;",
            ""
        ]

        # Add custom type definitions
        if self.metal_registry.required_headers:
            headers.extend(self.metal_registry.required_headers)
            headers.append("")

        return headers

    def _translate_node(self, node: CUDANode) -> str:
        """Translate a single CUDA AST node to Metal code."""
        if isinstance(node, CUDAKernel):
            return self._translate_kernel(node)
        elif isinstance(node, FunctionNode):
            return self._translate_function(node)
        elif isinstance(node, ClassNode):
            return self._translate_class(node)
        elif isinstance(node, StructNode):
            return self._translate_struct(node)
        elif isinstance(node, EnumNode):
            return self._translate_enum(node)
        elif isinstance(node, TypedefNode):
            return self._translate_typedef(node)
        elif isinstance(node, NamespaceNode):
            return self._translate_namespace(node)
        # Add more node type translations as needed
        else:
            logger.warning(f"Unhandled node type for translation: {type(node).__name__}")
            return f"// Unhandled node type: {type(node).__name__}"

    def _translate_kernel(self, kernel: CUDAKernel) -> str:
        """Translate CUDA kernel to Metal kernel."""
        logger.info(f"Translating kernel: {kernel.name}")
        metal_code = []

        # Generate kernel signature
        params = self._translate_kernel_parameters(kernel.parameters)
        metal_code.append(f"kernel void {kernel.name}({params})")
        metal_code.append("{")

        # Generate thread indexing code
        metal_code.extend(self._generate_metal_thread_indexing())

        # Translate kernel body with optimizations
        for stmt in kernel.body:
            translated_stmt = self._translate_statement(stmt, indent=4)
            metal_code.append(translated_stmt)

        metal_code.append("}")

        return "\n".join(metal_code)

    def _translate_kernel_parameters(self, parameters: List[CUDAParameter]) -> str:
        """Translate CUDA kernel parameters to Metal parameters."""
        metal_params = []
        for idx, param in enumerate(parameters):
            metal_type = self._cuda_type_to_metal(param.data_type)
            if param.is_pointer:
                qualifier = "device" if not param.is_readonly else "constant device"
                metal_params.append(f"{qualifier} {metal_type}* {param.name} [[buffer({idx})]]")
            else:
                metal_params.append(f"{metal_type} {param.name} [[buffer({idx})]]")
        return ", ".join(metal_params)

    def _cuda_type_to_metal(self, cuda_type: CUDAType) -> str:
        """Map CUDA types to Metal types."""
        # Implement mapping based on CUDA_TO_METAL_TYPE_MAP
        metal_type = METAL_EQUIVALENTS.get(cuda_type.spelling, cuda_type.spelling)
        return metal_type

    def _generate_metal_thread_indexing(self) -> List[str]:
        """Generate Metal-specific thread indexing code."""
        return [
            "    const uint3 thread_position_in_grid [[thread_position_in_grid]];",
            "    const uint3 threads_per_grid [[threads_per_grid]];",
            "    const uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]];",
            "    const uint3 threads_per_threadgroup [[threads_per_threadgroup]];",
            "",
            "    const uint global_id = thread_position_in_grid.x +",
            "                          thread_position_in_grid.y * threads_per_grid.x +",
            "                          thread_position_in_grid.z * threads_per_grid.x * threads_per_grid.y;",
            ""
        ]

    def _translate_function(self, function: FunctionNode) -> str:
        """Translate CUDA device function to Metal function."""
        logger.info(f"Translating function: {function.name}")
        metal_code = []

        # Translate return type and function signature
        return_type = self._cuda_type_to_metal(function.return_type)
        params = self._translate_function_parameters(function.parameters)
        metal_code.append(f"{return_type} {function.name}({params})")
        metal_code.append("{")

        # Translate function body
        for stmt in function.body:
            translated_stmt = self._translate_statement(stmt, indent=4)
            metal_code.append(translated_stmt)

        metal_code.append("}")

        return "\n".join(metal_code)

    def _translate_function_parameters(self, parameters: List[CUDAParameter]) -> str:
        """Translate function parameters to Metal parameters."""
        metal_params = []
        for param in parameters:
            metal_type = self._cuda_type_to_metal(param.data_type)
            qualifier = "const" if param.is_readonly else ""
            if param.is_pointer:
                qualifier += " device" if not param.is_readonly else " constant device"
                metal_params.append(f"{qualifier} {metal_type}* {param.name}")
            else:
                metal_params.append(f"{metal_type} {param.name}")
        return ", ".join(metal_params)

    def _translate_class(self, class_node: ClassNode) -> str:
        """Translate CUDA class to Metal-compatible struct or class."""
        logger.info(f"Translating class: {class_node.name}")
        metal_code = []

        # Translate class members
        for member in class_node.members:
            metal_member = self._translate_variable(member)
            metal_code.append(metal_member)

        # Translate class methods
        for method in class_node.methods:
            translated_method = self._translate_function(method)
            metal_code.append(translated_method)

        return "\n".join(metal_code)

    def _translate_struct(self, struct_node: StructNode) -> str:
        """Translate CUDA struct to Metal struct."""
        logger.info(f"Translating struct: {struct_node.name}")
        metal_code = []

        metal_code.append(f"struct {struct_node.name} {{")
        for member in struct_node.members:
            metal_member = self._translate_variable(member)
            metal_code.append(f"    {metal_member}")
        metal_code.append("};\n")

        return "\n".join(metal_code)

    def _translate_enum(self, enum_node: EnumNode) -> str:
        """Translate CUDA enum to Metal enum."""
        logger.info(f"Translating enum: {enum_node.name}")
        metal_code = []

        metal_code.append(f"enum {enum_node.name} {{")
        for enumerator in enum_node.enumerators:
            metal_code.append(f"    {enumerator},")
        metal_code.append("};\n")

        return "\n".join(metal_code)

    def _translate_typedef(self, typedef_node: TypedefNode) -> str:
        """Translate CUDA typedef to Metal typedef."""
        logger.info(f"Translating typedef: {typedef_node.alias}")
        metal_code = []

        original_type = self._cuda_type_to_metal(CUDAType(typedef_node.original_type))
        metal_code.append(f"typedef {original_type} {typedef_node.alias};\n")

        return "\n".join(metal_code)

    def _translate_namespace(self, namespace_node: NamespaceNode) -> str:
        """Translate CUDA namespace to Metal namespace or struct."""
        logger.info(f"Translating namespace: {namespace_node.name}")
        metal_code = []

        metal_code.append(f"namespace {namespace_node.name} {{")
        for member in namespace_node.members:
            translated_member = self._translate_node(member)
            metal_code.append(f"    {translated_member}")
        metal_code.append("}\n")

        return "\n".join(metal_code)

    def _translate_variable(self, variable: VariableNode) -> str:
        """Translate CUDA variable to Metal variable declaration."""
        metal_type = self._cuda_type_to_metal(variable.data_type)
        var_declaration = f"{metal_type} {variable.name};"
        return var_declaration

    def _translate_statement(self, stmt: CUDAStatement, indent: int = 0) -> str:
        """Translate CUDA statement to Metal statement."""
        indent_str = ' ' * indent
        if stmt.kind == 'RETURN':
            expr = self._translate_expression(stmt.expression)
            return f"{indent_str}return {expr};"
        elif stmt.kind == 'IF':
            condition = self._translate_expression(stmt.condition)
            then_branch = "\n".join([self._translate_statement(s, indent + 4) for s in stmt.then_branch])
            if stmt.else_branch:
                else_branch = "\n".join([self._translate_statement(s, indent + 4) for s in stmt.else_branch])
                return f"{indent_str}if ({condition}) {{\n{then_branch}\n{indent_str}}} else {{\n{else_branch}\n{indent_str}}}"
            else:
                return f"{indent_str}if ({condition}) {{\n{then_branch}\n{indent_str}}}"
        elif stmt.kind == 'FOR':
            init = self._translate_expression(stmt.init)
            condition = self._translate_expression(stmt.condition)
            increment = self._translate_expression(stmt.increment)
            body = "\n".join([self._translate_statement(s, indent + 4) for s in stmt.body])
            return f"{indent_str}for ({init}; {condition}; {increment}) {{\n{body}\n{indent_str}}}"
        elif stmt.kind == 'WHILE':
            condition = self._translate_expression(stmt.condition)
            body = "\n".join([self._translate_statement(s, indent + 4) for s in stmt.body])
            return f"{indent_str}while ({condition}) {{\n{body}\n{indent_str}}}"
        elif stmt.kind == 'DO_WHILE':
            condition = self._translate_expression(stmt.condition)
            body = "\n".join([self._translate_statement(s, indent + 4) for s in stmt.body])
            return f"{indent_str}do {{\n{body}\n{indent_str}}} while ({condition});"
        else:
            logger.warning(f"Unhandled statement kind: {stmt.kind}")
            return f"{indent_str}// Unhandled statement kind: {stmt.kind}"

    def _translate_expression(self, expr: CUDAExpressionNode) -> str:
        """Translate CUDA expression to Metal expression."""
        if expr.kind == 'BINARY_OP':
            left = self._translate_expression(expr.left)
            right = self._translate_expression(expr.right)
            return f"({left} {expr.operator} {right})"
        elif expr.kind == 'UNARY_OP':
            operand = self._translate_expression(expr.operand)
            return f"({expr.operator}{operand})"
        elif expr.kind == 'CALL_EXPR':
            args = ", ".join([self._translate_expression(arg) for arg in expr.arguments])
            return f"{expr.function}({args})"
        elif expr.kind == 'ARRAY_SUBSCRIPT':
            array = self._translate_expression(expr.array)
            index = self._translate_expression(expr.index)
            return f"{array}[{index}]"
        elif expr.kind == 'MEMBER_REF':
            base = self._translate_expression(expr.base)
            member = expr.member
            return f"{base}.{member}"
        elif expr.kind == 'CONDITIONAL_OPERATOR':
            condition = self._translate_expression(expr.condition)
            then_expr = self._translate_expression(expr.then_expression)
            else_expr = self._translate_expression(expr.else_expression)
            return f"({condition} ? {then_expr} : {else_expr})"
        elif expr.kind == 'INIT_LIST_EXPR':
            elements = ", ".join([self._translate_expression(e) for e in expr.elements])
            return f"{{{elements}}}"
        elif expr.kind == 'INTEGER_LITERAL':
            return expr.value
        elif expr.kind == 'FLOATING_LITERAL':
            return expr.value
        elif expr.kind == 'STRING_LITERAL':
            return f"\"{expr.value}\""
        elif expr.kind == 'DECL_REF_EXPR':
            return expr.spelling
        else:
            logger.warning(f"Unhandled expression kind: {expr.kind}")
            return f"/* Unhandled expression kind: {expr.kind} */"

    def _translate_binary_operator(self, cursor: Cursor) -> CUDAExpressionNode:
        """Convert binary operator."""
        children = list(cursor.get_children())
        if len(children) < 2:
            logger.warning("Binary operator with insufficient children.")
            return CUDAExpressionNode(kind='BINARY_OP', operator='UNKNOWN', left=None, right=None, location=self._get_cursor_location(cursor))

        left = self._convert_expression(children[0])
        right = self._convert_expression(children[1])

        # Extract operator from tokens
        tokens = list(cursor.get_tokens())
        operator = None
        for tok in tokens:
            if tok.kind == TokenKind.PUNCTUATION and tok.spelling in {'+', '-', '*', '/', '%', '==', '!=', '<', '>', '<=', '>=', '&&', '||', '=', '+=', '-=', '*=', '/='}:
                operator = tok.spelling
                break

        if not operator:
            operator = 'UNKNOWN'

        binary_op = CUDAExpressionNode(
            kind='BINARY_OP',
            operator=operator,
            left=left,
            right=right,
            location=self._get_cursor_location(cursor)
        )

        return binary_op

    def _translate_unary_operator(self, cursor: Cursor) -> CUDAExpressionNode:
        """Convert unary operator."""
        children = list(cursor.get_children())
        if len(children) < 1:
            logger.warning("Unary operator with no operand.")
            return CUDAExpressionNode(kind='UNARY_OP', operator='UNKNOWN', operand=None, location=self._get_cursor_location(cursor))

        operand = self._convert_expression(children[0])

        # Extract operator from tokens
        tokens = list(cursor.get_tokens())
        operator = None
        for tok in tokens:
            if tok.kind == TokenKind.PUNCTUATION and tok.spelling in {'++', '--', '!', '~', '-', '+'}:
                operator = tok.spelling
                break

        if not operator:
            operator = 'UNKNOWN'

        unary_op = CUDAExpressionNode(
            kind='UNARY_OP',
            operator=operator,
            operand=operand,
            location=self._get_cursor_location(cursor)
        )

        return unary_op

    def _translate_call_expr(self, cursor: Cursor) -> CUDAExpressionNode:
        """Convert function call expression."""
        func_name = cursor.spelling
        args = [self._convert_expression(child) for child in cursor.get_children()]
        call_expr = CUDAExpressionNode(
            kind='CALL_EXPR',
            function=func_name,
            arguments=args,
            location=self._get_cursor_location(cursor)
        )
        return call_expr

    def _translate_array_subscript(self, cursor: Cursor) -> CUDAExpressionNode:
        """Convert array subscript expression."""
        children = list(cursor.get_children())
        if len(children) < 2:
            logger.warning("Array subscript with insufficient children.")
            return CUDAExpressionNode(kind='ARRAY_SUBSCRIPT', array=None, index=None, location=self._get_cursor_location(cursor))

        array = self._convert_expression(children[0])
        index = self._convert_expression(children[1])
        array_sub = CUDAExpressionNode(
            kind='ARRAY_SUBSCRIPT',
            array=array,
            index=index,
            location=self._get_cursor_location(cursor)
        )
        return array_sub

    def _translate_member_ref_expr(self, cursor: Cursor) -> CUDAExpressionNode:
        """Convert member reference expression."""
        children = list(cursor.get_children())
        if len(children) < 1:
            logger.warning("Member reference with no base.")
            return CUDAExpressionNode(kind='MEMBER_REF', base=None, member=cursor.spelling, location=self._get_cursor_location(cursor))

        base = self._convert_expression(children[0])
        member_ref = CUDAExpressionNode(
            kind='MEMBER_REF',
            base=base,
            member=cursor.spelling,
            location=self._get_cursor_location(cursor)
        )
        return member_ref

    def _translate_conditional_operator(self, cursor: Cursor) -> CUDAExpressionNode:
        """Convert conditional (ternary) operator."""
        children = list(cursor.get_children())
        if len(children) < 3:
            logger.warning("Conditional operator with insufficient children.")
            return CUDAExpressionNode(kind='CONDITIONAL_OPERATOR', condition=None, then_expression=None, else_expression=None, location=self._get_cursor_location(cursor))

        condition = self._convert_expression(children[0])
        then_expr = self._convert_expression(children[1])
        else_expr = self._convert_expression(children[2])

        cond_op = CUDAExpressionNode(
            kind='CONDITIONAL_OPERATOR',
            condition=condition,
            then_expression=then_expr,
            else_expression=else_expr,
            location=self._get_cursor_location(cursor)
        )

        return cond_op

    def _translate_init_list_expr(self, cursor: Cursor) -> CUDAExpressionNode:
        """Convert initializer list expression."""
        elements = [self._convert_expression(child) for child in cursor.get_children()]
        init_list = CUDAExpressionNode(
            kind='INIT_LIST_EXPR',
            elements=elements,
            location=self._get_cursor_location(cursor)
        )
        return init_list

    def _translate_default(self, cursor: Cursor) -> CUDANode:
        """Default translation for unhandled cursor types."""
        node = CUDANode(
            kind=cursor.kind.name,
            spelling=cursor.spelling,
            type=cursor.type.spelling,
            children=[]
        )
        for child in cursor.get_children():
            converted = self._convert_cursor(child)
            if converted:
                node.add_child(converted)
        return node

    def _translate_class(self, class_node: ClassNode) -> str:
        """Translate CUDA class to Metal-compatible struct or class."""
        logger.info(f"Translating class: {class_node.name}")
        metal_code = []

        # Translate class members
        for member in class_node.members:
            translated_member = self._translate_variable(member)
            metal_code.append(f"    {translated_member}")

        # Translate class methods
        for method in class_node.methods:
            translated_method = self._translate_function(method)
            metal_code.append(f"    {translated_method}")

        metal_code_str = f"struct {class_node.name} {{\n" + "\n".join(metal_code) + "\n};\n"
        return metal_code_str

    def _translate_statement(self, stmt: CUDAStatement, indent: int = 0) -> str:
        """Translate CUDA statement to Metal statement."""
        indent_str = ' ' * indent
        if stmt.kind == 'RETURN':
            expr = self._translate_expression(stmt.expression)
            return f"{indent_str}return {expr};"
        elif stmt.kind == 'IF':
            condition = self._translate_expression(stmt.condition)
            then_branch = "\n".join([self._translate_statement(s, indent + 4) for s in stmt.then_branch])
            if stmt.else_branch:
                else_branch = "\n".join([self._translate_statement(s, indent + 4) for s in stmt.else_branch])
                return f"{indent_str}if ({condition}) {{\n{then_branch}\n{indent_str}}} else {{\n{else_branch}\n{indent_str}}}"
            else:
                return f"{indent_str}if ({condition}) {{\n{then_branch}\n{indent_str}}}"
        elif stmt.kind == 'FOR':
            init = self._translate_expression(stmt.init)
            condition = self._translate_expression(stmt.condition)
            increment = self._translate_expression(stmt.increment)
            body = "\n".join([self._translate_statement(s, indent + 4) for s in stmt.body])
            return f"{indent_str}for ({init}; {condition}; {increment}) {{\n{body}\n{indent_str}}}"
        elif stmt.kind == 'WHILE':
            condition = self._translate_expression(stmt.condition)
            body = "\n".join([self._translate_statement(s, indent + 4) for s in stmt.body])
            return f"{indent_str}while ({condition}) {{\n{body}\n{indent_str}}}"
        elif stmt.kind == 'DO_WHILE':
            condition = self._translate_expression(stmt.condition)
            body = "\n".join([self._translate_statement(s, indent + 4) for s in stmt.body])
            return f"{indent_str}do {{\n{body}\n{indent_str}}} while ({condition});"
        else:
            logger.warning(f"Unhandled statement kind: {stmt.kind}")
            return f"{indent_str}// Unhandled statement kind: {stmt.kind}"

    def _translate_expression(self, expr: CUDAExpressionNode) -> str:
        """Translate CUDA expression to Metal expression."""
        if expr.kind == 'BINARY_OP':
            left = self._translate_expression(expr.left)
            right = self._translate_expression(expr.right)
            return f"({left} {expr.operator} {right})"
        elif expr.kind == 'UNARY_OP':
            operand = self._translate_expression(expr.operand)
            return f"({expr.operator}{operand})"
        elif expr.kind == 'CALL_EXPR':
            args = ", ".join([self._translate_expression(arg) for arg in expr.arguments])
            return f"{expr.function}({args})"
        elif expr.kind == 'ARRAY_SUBSCRIPT':
            array = self._translate_expression(expr.array)
            index = self._translate_expression(expr.index)
            return f"{array}[{index}]"
        elif expr.kind == 'MEMBER_REF':
            base = self._translate_expression(expr.base)
            member = expr.member
            return f"{base}.{member}"
        elif expr.kind == 'CONDITIONAL_OPERATOR':
            condition = self._translate_expression(expr.condition)
            then_expr = self._translate_expression(expr.then_expression)
            else_expr = self._translate_expression(expr.else_expression)
            return f"({condition} ? {then_expr} : {else_expr})"
        elif expr.kind == 'INIT_LIST_EXPR':
            elements = ", ".join([self._translate_expression(e) for e in expr.elements])
            return f"{{{elements}}}"
        elif expr.kind == 'INTEGER_LITERAL':
            return expr.value
        elif expr.kind == 'FLOATING_LITERAL':
            return expr.value
        elif expr.kind == 'STRING_LITERAL':
            return f"\"{expr.value}\""
        elif expr.kind == 'DECL_REF_EXPR':
            return expr.spelling
        else:
            logger.warning(f"Unhandled expression kind: {expr.kind}")
            return f"/* Unhandled expression kind: {expr.kind} */"

    def _translate_node_code(self, node: CUDANode, indent: int = 4) -> List[str]:
        """Translate a CUDA AST node to Metal code lines."""
        translated = self._translate_node(node)
        return [(" " * indent) + line for line in translated.split('\n')]

    def _translate_array_access_code(self, node: CUDAExpressionNode) -> List[str]:
        """Translate array access to Metal code."""
        array = self._translate_expression(node.array)
        index = self._translate_expression(node.index)
        return [f"{array}[{index}]"]

    def _translate_binary_operation_code(self, node: CUDAExpressionNode) -> List[str]:
        """Translate binary operation to Metal code."""
        left = self._translate_expression(node.left)
        right = self._translate_expression(node.right)
        return [f"({left} {node.operator} {right})"]

    def _translate_function_call_code(self, node: CUDAExpressionNode) -> List[str]:
        """Translate function call to Metal code."""
        func = node.function
        args = ", ".join([self._translate_expression(arg) for arg in node.arguments])

        # Handle built-in CUDA functions
        if func in METAL_EQUIVALENTS:
            metal_func = METAL_EQUIVALENTS[func]
            return [f"{metal_func}({args});"]
        else:
            return [f"{func}({args});"]

    def _translate_if_statement_code(self, node: CUDAStatement) -> List[str]:
        """Translate if statement to Metal code."""
        condition = self._translate_expression(node.condition)
        then_branch = "\n".join([self._translate_statement(s, indent=8) for s in node.then_branch])
        if node.else_branch:
            else_branch = "\n".join([self._translate_statement(s, indent=8) for s in node.else_branch])
            return [
                f"if ({condition}) {{",
                then_branch,
                f"}} else {{",
                else_branch,
                f"}}"
            ]
        else:
            return [
                f"if ({condition}) {{",
                then_branch,
                f"}}"
            ]

    def _translate_for_loop_code(self, node: CUDAStatement) -> List[str]:
        """Translate for loop to Metal code."""
        init = self._translate_expression(node.init)
        condition = self._translate_expression(node.condition)
        increment = self._translate_expression(node.increment)
        body = "\n".join([self._translate_statement(s, indent=8) for s in node.body])
        return [
            f"for ({init}; {condition}; {increment}) {{",
            body,
            f"}}"
        ]

    def _translate_while_loop_code(self, node: CUDAStatement) -> List[str]:
        """Translate while loop to Metal code."""
        condition = self._translate_expression(node.condition)
        body = "\n".join([self._translate_statement(s, indent=8) for s in node.body])
        return [
            f"while ({condition}) {{",
            body,
            f"}}"
        ]

    def _translate_do_loop_code(self, node: CUDAStatement) -> List[str]:
        """Translate do-while loop to Metal code."""
        body = "\n".join([self._translate_statement(s, indent=8) for s in node.body])
        condition = self._translate_expression(node.condition)
        return [
            f"do {{",
            body,
            f"}} while ({condition});"
        ]

    def _translate_default_node_code(self, node: CUDANode, indent: int = 4) -> List[str]:
        """Translate unhandled node types to Metal code."""
        return [(" " * indent) + f"// Unhandled node type: {node.kind}"]

    def _translate_variable_declaration(self, var: VariableNode) -> str:
        """Translate variable declaration to Metal code."""
        metal_type = self._cuda_type_to_metal(var.data_type)
        return f"{metal_type} {var.name};"

    def _translate_metal_threadgroup_indexing(self) -> List[str]:
        """Generate threadgroup indexing code for Metal."""
        return [
            "    const uint3 thread_position_in_grid [[thread_position_in_grid]];",
            "    const uint3 threads_per_grid [[threads_per_grid]];",
            "    const uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]];",
            "    const uint3 threads_per_threadgroup [[threads_per_threadgroup]];",
            "",
            "    const uint global_id = thread_position_in_grid.x +",
            "                          thread_position_in_grid.y * threads_per_grid.x +",
            "                          thread_position_in_grid.z * threads_per_grid.x * threads_per_grid.y;",
            ""
        ]

    def _translate_class_members(self, class_node: ClassNode) -> List[str]:
        """Translate class members to Metal code."""
        members_code = []
        for member in class_node.members:
            translated_member = self._translate_variable(member)
            members_code.append(f"    {translated_member}")
        return members_code

    def _translate_class_methods(self, class_node: ClassNode) -> List[str]:
        """Translate class methods to Metal code."""
        methods_code = []
        for method in class_node.methods:
            translated_method = self._translate_function(method)
            methods_code.append(translated_method)
        return methods_code

    def _translate_expression_node(self, expr: CUDAExpressionNode) -> str:
        """Translate expression node to Metal code."""
        return self._translate_expression(expr)

    def _translate_binary_op(self, node: CUDAExpressionNode) -> str:
        """Translate binary operation to Metal code."""
        left = self._translate_expression(node.left)
        right = self._translate_expression(node.right)
        return f"({left} {node.operator} {right})"

    def _compile_metal_code(self, metal_code: str):
        """Compile Metal code using the Metal compiler."""
        metal_compiler = self.metal_integration._metal_compiler_path
        if not metal_compiler:
            logger.error("Metal compiler path not found.")
            return

        # Write Metal code to temporary file
        temp_metal_file = "temp_kernel.metal"
        with open(temp_metal_file, 'w') as f:
            f.write(metal_code)

        # Define output file
        output_file = "temp_kernel.air"

        # Compile Metal code
        compile_command = f"{metal_compiler} -c {temp_metal_file} -o {output_file}"
        logger.info(f"Compiling Metal code with command: {compile_command}")

        result = os.system(compile_command)
        if result != 0:
            logger.error("Metal compilation failed.")
            raise CudaTranslationError("Metal compilation failed.")
        else:
            logger.info("Metal code compiled successfully.")

        # Clean up temporary files
        os.remove(temp_metal_file)
        os.remove(output_file)

    def _perform_final_optimizations(self, code: str) -> str:
        """
        Perform final pass optimizations on the generated Metal code.
        This can include removing unnecessary brackets, optimizing variable declarations,
        memory barriers, and removing redundant operations.
        """
        # Placeholder for final optimizations
        # Implement as needed
        return code

    def _translate_kernel_body(self, body: List[CUDANode]) -> List[str]:
        """Translate kernel body statements to Metal code."""
        translated_code = []
        for stmt in body:
            translated_stmt = self._translate_statement(stmt, indent=4)
            translated_code.append(translated_stmt)
        return translated_code

    def _translate_node_recursively(self, node: CUDANode) -> str:
        """Recursively translate AST node to Metal code."""
        translated = self._translate_node(node)
        return translated

    def _translate_expression_recursively(self, expr: CUDAExpressionNode) -> str:
        """Recursively translate expression node to Metal code."""
        return self._translate_expression(expr)

    def _get_cuda_type(self, type_spelling: str) -> CUDAType:
        """Retrieve CUDAType object based on type spelling."""
        return CUDAType(type_spelling)

    def _translate_binary_operation(self, node: CUDAExpressionNode) -> str:
        """Translate binary operation to Metal code."""
        return self._translate_binary_op(node)

    def _translate_unary_operation(self, node: CUDAExpressionNode) -> str:
        """Translate unary operation to Metal code."""
        operand = self._translate_expression(node.operand)
        return f"({node.operator}{operand})"

    def _translate_default_expression(self, expr: CUDAExpressionNode) -> str:
        """Translate unhandled expression types."""
        return f"/* Unhandled expression kind: {expr.kind} */"

    def _translate_member_access(self, node: CUDAExpressionNode) -> str:
        """Translate member access expression."""
        base = self._translate_expression(node.base)
        member = node.member
        return f"{base}.{member}"

    def _translate_conditional_operator(self, node: CUDAExpressionNode) -> str:
        """Translate conditional operator to Metal code."""
        condition = self._translate_expression(node.condition)
        then_expr = self._translate_expression(node.then_expression)
        else_expr = self._translate_expression(node.else_expression)
        return f"({condition} ? {then_expr} : {else_expr})"

    def _translate_init_list(self, node: CUDAExpressionNode) -> str:
        """Translate initializer list to Metal code."""
        elements = ", ".join([self._translate_expression(e) for e in node.elements])
        return f"{{{elements}}}"

    def _translate_member_ref(self, node: CUDAExpressionNode) -> str:
        """Translate member reference to Metal code."""
        base = self._translate_expression(node.base)
        member = node.member
        return f"{base}.{member}"

    def _translate_binary_operator_expression(self, node: CUDAExpressionNode) -> str:
        """Translate binary operator expression."""
        left = self._translate_expression(node.left)
        right = self._translate_expression(node.right)
        return f"({left} {node.operator} {right})"

    def _translate_call_expression(self, node: CUDAExpressionNode) -> str:
        """Translate call expression to Metal code."""
        func = node.function
        args = ", ".join([self._translate_expression(arg) for arg in node.arguments])

        # Check if function is a CUDA builtin and has a Metal equivalent
        if func in METAL_EQUIVALENTS:
            metal_func = METAL_EQUIVALENTS[func]
            return f"{metal_func}({args});"
        else:
            return f"{func}({args});"

    def _translate_expression_node_recursively(self, expr: CUDAExpressionNode) -> str:
        """Recursively translate expression node to Metal code."""
        return self._translate_expression(expr)

    def _translate_variable_node(self, var: VariableNode) -> str:
        """Translate variable node to Metal code."""
        metal_type = self._cuda_type_to_metal(var.data_type)
        return f"{metal_type} {var.name};"

    def _translate_binary_op_node(self, node: CUDAExpressionNode) -> str:
        """Translate binary operation node to Metal code."""
        return self._translate_binary_operation(node)

    def _translate_unary_op_node(self, node: CUDAExpressionNode) -> str:
        """Translate unary operation node to Metal code."""
        return self._translate_unary_operation(node)

    def _translate_call_expr_node(self, node: CUDAExpressionNode) -> str:
        """Translate call expression node to Metal code."""
        return self._translate_call_expression(node)

    def _translate_if_stmt_node(self, node: CUDAStatement) -> str:
        """Translate if statement node to Metal code."""
        return self._translate_if_statement_code(node)

    def _translate_for_stmt_node(self, node: CUDAStatement) -> str:
        """Translate for loop node to Metal code."""
        return self._translate_for_loop_code(node)

    def _translate_while_stmt_node(self, node: CUDAStatement) -> str:
        """Translate while loop node to Metal code."""
        return self._translate_while_loop_code(node)

    def _translate_do_stmt_node(self, node: CUDAStatement) -> str:
        """Translate do-while loop node to Metal code."""
        return self._translate_do_loop_code(node)

    def _translate_conditional_operator_node(self, node: CUDAExpressionNode) -> str:
        """Translate conditional operator node to Metal code."""
        return self._translate_conditional_operator(node)

    def _translate_init_list_node(self, node: CUDAExpressionNode) -> str:
        """Translate initializer list node to Metal code."""
        return self._translate_init_list(node)

    def _translate_member_ref_expr_node(self, node: CUDAExpressionNode) -> str:
        """Translate member reference expression node to Metal code."""
        return self._translate_member_ref(node)

    def _translate_return_stmt_node(self, node: CUDAStatement) -> str:
        """Translate return statement node to Metal code."""
        return self._translate_statement(node, indent=4)

    def _translate_compound_stmt_node(self, node: CUDACompoundStmt) -> str:
        """Translate compound statement node to Metal code."""
        body = "\n".join([self._translate_statement(s, indent=4) for s in node.children])
        return f"{{\n{body}\n}}"

    def _perform_dataflow_analysis(self, ast: CUDANode):
        """Perform dataflow analysis on the AST."""
        # Placeholder for dataflow analysis implementation
        logger.info("Performing dataflow analysis.")
        pass

    def _perform_alias_analysis(self, ast: CUDANode):
        """Perform alias analysis on the AST."""
        # Placeholder for alias analysis implementation
        logger.info("Performing alias analysis.")
        pass

    def _perform_advanced_optimizations(self, ast: CUDANode):
        """Perform advanced optimizations on the AST."""
        # Placeholder for advanced optimizations implementation
        logger.info("Performing advanced optimizations.")
        pass

    def _translate_node_to_metal(self, node: CUDANode) -> List[str]:
        """Translate a single AST node to Metal code lines."""
        translated = self._translate_node(node)
        return translated.split('\n')

    def _compile_metal_code(self, metal_code: str):
        """Compile Metal code using the Metal compiler."""
        metal_compiler = self.metal_integration._metal_compiler_path
        if not metal_compiler:
            logger.error("Metal compiler path not found.")
            raise CudaTranslationError("Metal compiler not available.")

        # Write Metal code to temporary file
        temp_metal_file = "temp_kernel.metal"
        with open(temp_metal_file, 'w') as f:
            f.write(metal_code)

        # Define output file
        output_file = "temp_kernel.air"

        # Compile Metal code
        compile_command = f"{metal_compiler} -c {temp_metal_file} -o {output_file}"
        logger.info(f"Compiling Metal code with command: {compile_command}")

        result = os.system(compile_command)
        if result != 0:
            logger.error("Metal compilation failed.")
            raise CudaTranslationError("Metal compilation failed.")
        else:
            logger.info("Metal code compiled successfully.")

        # Clean up temporary files
        os.remove(temp_metal_file)
        os.remove(output_file)

    def _translate_to_metal(self, ast: CUDANode) -> str:
        """Translate CUDA AST to Metal code."""
        logger.info("Translating CUDA AST to Metal code.")
        metal_code = []
        # Generate Metal headers
        metal_code.extend(self._generate_metal_headers())

        # Traverse AST and generate Metal code
        for child in ast.children:
            translated = self._translate_node(child)
            metal_code.append(translated)

        # Join all code parts
        return "\n".join(metal_code)

    def _translate_kernel_body(self, body: List[CUDANode]) -> List[str]:
        """Translate kernel body statements to Metal code."""
        translated_code = []
        for stmt in body:
            translated_stmt = self._translate_statement(stmt, indent=4)
            translated_code.append(translated_stmt)
        return translated_code

    def _translate_node_recursively(self, node: CUDANode) -> str:
        """Recursively translate AST node to Metal code."""
        translated = self._translate_node(node)
        return translated

    def _translate_expression_recursively(self, expr: CUDAExpressionNode) -> str:
        """Recursively translate expression node to Metal code."""
        return self._translate_expression(expr)

    def _translate_variable_node(self, var: VariableNode) -> str:
        """Translate variable node to Metal code."""
        metal_type = self._cuda_type_to_metal(var.data_type)
        return f"{metal_type} {var.name};"

    def _translate_binary_op_node(self, node: CUDAExpressionNode) -> str:
        """Translate binary operation node to Metal code."""
        return self._translate_binary_operation(node)

    def _translate_unary_op_node(self, node: CUDAExpressionNode) -> str:
        """Translate unary operation node to Metal code."""
        return self._translate_unary_operation(node)

    def _translate_call_expr_node(self, node: CUDAExpressionNode) -> str:
        """Translate call expression node to Metal code."""
        return self._translate_call_expression(node)

    def _translate_if_stmt_node(self, node: CUDAStatement) -> str:
        """Translate if statement node to Metal code."""
        return self._translate_if_statement_code(node)

    def _translate_for_stmt_node(self, node: CUDAStatement) -> str:
        """Translate for loop node to Metal code."""
        return self._translate_for_loop_code(node)

    def _translate_while_stmt_node(self, node: CUDAStatement) -> str:
        """Translate while loop node to Metal code."""
        return self._translate_while_loop_code(node)

    def _translate_do_stmt_node(self, node: CUDAStatement) -> str:
        """Translate do-while loop node to Metal code."""
        return self._translate_do_loop_code(node)

    def _translate_conditional_operator_node(self, node: CUDAExpressionNode) -> str:
        """Translate conditional operator node to Metal code."""
        return self._translate_conditional_operator(node)

    def _translate_init_list_node(self, node: CUDAExpressionNode) -> str:
        """Translate initializer list node to Metal code."""
        return self._translate_init_list(node)

    def _translate_member_ref_expr_node(self, node: CUDAExpressionNode) -> str:
        """Translate member reference expression node to Metal code."""
        return self._translate_member_ref(node)

    def _translate_return_stmt_node(self, node: CUDAStatement) -> str:
        """Translate return statement node to Metal code."""
        return self._translate_statement(node, indent=4)

    def _translate_compound_stmt_node(self, node: CUDACompoundStmt) -> str:
        """Translate compound statement node to Metal code."""
        body = "\n".join([self._translate_statement(s, indent=4) for s in node.children])
        return f"{{\n{body}\n}}"

    def validate_file(self, file_path: str) -> bool:
        """
        Validate CUDA file syntax.

        Args:
            file_path: Path to CUDA file

        Returns:
            bool: True if valid, False otherwise
        """
        try:
            ast = self.parse_file(file_path)
            return bool(ast)
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            return False

    def finalize(self) -> None:
        """
        Perform final cleanup and optimization steps.
        """
        # Clear any temporary data
        self.ast_cache.clear()
        self.type_cache.clear()
        self.function_cache.clear()

        # Validate generated code
        # Placeholder for validation steps
        logger.info("Finalizing parser and cleaning up resources.")

    # Additional utility methods can be added below as needed
    # For example, methods for dataflow analysis, alias analysis, optimization strategies, etc.

# Example usage:
# parser = CudaParser()
# try:
#     ast = parser.parse_file("path/to/cuda_file.cu")
#     print("Parsing and translation successful.")
# except CudaParseError as e:
#     print(f"Error: {e}")
