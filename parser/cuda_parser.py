
import clang.cindex
from clang.cindex import Index, TranslationUnit, Cursor, CursorKind, TypeKind
from typing import Dict, List, Optional, Union, Tuple, Any
from threading import Lock
import os
import platform
import logging
import hashlib
import time

from ..utils.error_handler import CudaParseError, CudaTranslationError
from ..utils.logger import get_logger
from ..core.parser.ast_nodes import (
    CUDANode, CUDAKernel, CUDAParameter, CUDAType, CUDAQualifier,
    CUDAExpressionNode, CUDAStatement, KernelNode, FunctionNode, VariableNode
)

logger = get_logger(__name__)

class CudaParser:

    def __init__(self, cuda_include_paths: Optional[List[str]] = None):
        # Initialize clang parser
        self.index = Index.create()
        self.cuda_include_paths = cuda_include_paths or self._find_cuda_paths()
        self._lock = Lock()
        self._ast_cache: Dict[str, CUDANode] = {}
        self._setup_clang_args()

    def _find_cuda_paths(self) -> List[str]:
        """Find CUDA installation paths based on platform."""
        cuda_paths = []

        # Common locations by platform
        if platform.system() == 'Windows':
            program_files = os.environ.get('ProgramFiles', 'C:\\Program Files')
            cuda_base = os.path.join(program_files, 'NVIDIA GPU Computing Toolkit', 'CUDA')
            if os.path.exists(cuda_base):
                for version_dir in sorted(os.listdir(cuda_base), reverse=True):
                    if version_dir.startswith('v'):
                        include_path = os.path.join(cuda_base, version_dir, 'include')
                        if os.path.exists(include_path):
                            cuda_paths.append(include_path)
                            break
        elif platform.system() == 'Linux':
            for path in ['/usr/local/cuda/include', '/usr/include/cuda']:
                if os.path.exists(path):
                    cuda_paths.append(path)
        elif platform.system() == 'Darwin':
            for path in ['/usr/local/cuda/include', '/opt/cuda/include']:
                if os.path.exists(path):
                    cuda_paths.append(path)

        # Add standard system paths
        cuda_paths.extend(['/usr/include', '/usr/local/include'])
        return cuda_paths

    def _setup_clang_args(self):
        """Set up clang compilation arguments for CUDA parsing."""
        self.clang_args = [
            '-x', 'cuda',                      # Treat input as CUDA source
            '--cuda-gpu-arch=sm_70',           # Target compute capability
            '-std=c++14',                      # C++ standard
            '-D__CUDACC__',                    # Define CUDACC preprocessor macro
            '-D__CUDA_ARCH__=700',             # Define CUDA architecture
            '-DNDEBUG',                        # Define NDEBUG for release mode
        ]

        # Add include paths
        for path in self.cuda_include_paths:
            self.clang_args.extend(['-I', path])

    def parse_file(self, file_path: str) -> Optional[CUDANode]:
        """
        Parse CUDA source file into AST with full error handling and caching.

        Args:
            file_path: Path to CUDA source file

        Returns:
            CUDANode: Root AST node

        Raises:
            CudaParseError: If parsing fails
            FileNotFoundError: If file doesn't exist
        """
        # Input validation
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CUDA source file not found: {file_path}")

        # Check cache using file hash
        file_hash = self._get_file_hash(file_path)
        cache_key = f"{file_path}:{file_hash}"

        with self._lock:
            if cache_key in self._ast_cache:
                logger.debug(f"Using cached AST for {file_path}")
                return self._ast_cache[cache_key]

        try:
            # Parse with clang
            start_time = time.time()
            logger.info(f"Parsing CUDA file: {file_path}")

            tu = self.index.parse(
                file_path,
                args=self.clang_args,
                options=(
                        TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD |
                        TranslationUnit.PARSE_INCOMPLETE
                )
            )

            # Check for fatal errors
            if self._has_fatal_errors(tu):
                return None

            # Convert to our AST
            root = self._process_translation_unit(tu.cursor)

            # Cache the result
            with self._lock:
                self._ast_cache[cache_key] = root

            parse_time = time.time() - start_time
            logger.info(f"Successfully parsed {file_path} in {parse_time:.2f}s")

            return root

        except Exception as e:
            logger.error(f"Failed to parse {file_path}: {str(e)}")
            raise CudaParseError(f"Parse error: {str(e)}")

    def _get_file_hash(self, file_path: str) -> str:
        """Generate hash for file content."""
        try:
            with open(file_path, 'rb') as f:
                file_content = f.read()
                return hashlib.md5(file_content).hexdigest()
        except Exception as e:
            logger.warning(f"Failed to hash file {file_path}: {str(e)}")
            # Fall back to modification time if hashing fails
            return str(os.path.getmtime(file_path))

    def _has_fatal_errors(self, tu: TranslationUnit) -> bool:
        """Check for fatal parsing errors."""
        has_fatal = False
        for diag in tu.diagnostics:
            if diag.severity >= diag.Error:
                logger.error(
                    f"{diag.location.file}:{diag.location.line} - {diag.spelling}"
                )
                has_fatal = True
        return has_fatal

    def _process_translation_unit(self, cursor: Cursor) -> CUDANode:
        """Process translation unit cursor into CUDANode."""
        root = CUDANode(
            kind=str(cursor.kind),
            spelling=cursor.spelling,
            line=cursor.location.line,
            column=cursor.location.column
        )

        # Process all top-level declarations
        for child in cursor.get_children():
            if child.location.file and child.location.file.name.endswith(('.cu', '.cuh')):
                node = self._process_cursor(child)
                if node:
                    root.add_child(node)

        return root

    def _process_cursor(self, cursor: Cursor) -> Optional[CUDANode]:
        """Process a cursor into the appropriate CUDA AST node."""
        try:
            # Identify cursor type
            if cursor.kind == CursorKind.FUNCTION_DECL:
                # Check if it's a CUDA __global__ function (kernel)
                if any(c.kind == CursorKind.CUDA_GLOBAL_ATTR for c in cursor.get_children()):
                    return self._process_kernel(cursor)
                else:
                    return self._process_function(cursor)
            elif cursor.kind == CursorKind.VAR_DECL:
                return self._process_variable(cursor)
            elif cursor.kind in (CursorKind.STRUCT_DECL, CursorKind.CLASS_DECL):
                return self._process_struct_or_class(cursor)
            elif cursor.kind == CursorKind.TYPEDEF_DECL:
                return self._process_typedef(cursor)

            # Other node types can be added as needed
            return None

        except Exception as e:
            logger.error(f"Error processing cursor {cursor.spelling}: {str(e)}")
            return None

    def _process_kernel(self, cursor: Cursor) -> KernelNode:
        """Process a CUDA kernel function."""
        # Extract location information
        location = {
            'file': cursor.location.file.name if cursor.location.file else '',
            'line': cursor.location.line,
            'column': cursor.location.column
        }

        # Process parameters
        parameters = []
        for child in cursor.get_arguments():
            param = self._process_parameter(child)
            if param:
                parameters.append(param)

        # Process function body
        body = []
        for child in cursor.get_children():
            if child.kind == CursorKind.COMPOUND_STMT:
                for stmt in child.get_children():
                    node = self._process_statement(stmt)
                    if node:
                        body.append(node)

        # Create kernel node
        kernel = KernelNode(
            name=cursor.spelling,
            return_type=self._get_type_spelling(cursor.result_type),
            parameters=parameters,
            body=body,
            line=location['line'],
            column=location['column']
        )

        # Set kernel-specific attributes
        kernel.is_kernel = True

        # Process any CUDA specific attributes
        for child in cursor.get_children():
            if child.kind == CursorKind.CUDA_GLOBAL_ATTR:
                self._process_kernel_attributes(child, kernel)

        return kernel

    def _process_function(self, cursor: Cursor) -> FunctionNode:
        """Process a regular CUDA function."""
        # Similar to _process_kernel but for non-kernel functions
        location = {
            'file': cursor.location.file.name if cursor.location.file else '',
            'line': cursor.location.line,
            'column': cursor.location.column
        }

        # Process parameters
        parameters = []
        for child in cursor.get_arguments():
            param = self._process_parameter(child)
            if param:
                parameters.append(param)

        # Process function body
        body = []
        for child in cursor.get_children():
            if child.kind == CursorKind.COMPOUND_STMT:
                for stmt in child.get_children():
                    node = self._process_statement(stmt)
                    if node:
                        body.append(node)

        # Create function node
        function = FunctionNode(
            name=cursor.spelling,
            return_type=self._get_type_spelling(cursor.result_type),
            parameters=parameters,
            body=body,
            line=location['line'],
            column=location['column']
        )

        # Check for device attribute
        function.is_device = any(c.kind == CursorKind.CUDA_DEVICE_ATTR for c in cursor.get_children())

        return function

    def _process_parameter(self, cursor: Cursor) -> CUDAParameter:
        """Process a function parameter."""
        return CUDAParameter(
            name=cursor.spelling,
            param_type=self._get_type_spelling(cursor.type),
            is_pointer=cursor.type.kind == TypeKind.POINTER,
            qualifiers=self._get_qualifiers(cursor),
            line=cursor.location.line,
            column=cursor.location.column
        )

    def _process_variable(self, cursor: Cursor) -> VariableNode:
        """Process a variable declaration."""
        # Check if it's a special memory space variable
        is_shared = any(c.kind == CursorKind.CUDA_SHARED_ATTR for c in cursor.get_children())
        is_constant = any(c.kind == CursorKind.CUDA_CONSTANT_ATTR for c in cursor.get_children())
        is_device = any(c.kind == CursorKind.CUDA_DEVICE_ATTR for c in cursor.get_children())

        qualifiers = []
        if is_shared:
            qualifiers.append(CUDAQualifier.SHARED)
        if is_constant:
            qualifiers.append(CUDAQualifier.CONST)
        if is_device:
            qualifiers.append(CUDAQualifier.DEVICE)

        return VariableNode(
            name=cursor.spelling,
            var_type=self._get_type_spelling(cursor.type),
            qualifiers=qualifiers,
            is_pointer=cursor.type.kind == TypeKind.POINTER,
            line=cursor.location.line,
            column=cursor.location.column
        )

    def _process_statement(self, cursor: Cursor) -> Optional[CUDAStatement]:
        """Process a statement in the function body."""
        stmt_kind = str(cursor.kind)

        # Basic statement processing - can be expanded with more detailed processing
        if cursor.kind == CursorKind.COMPOUND_STMT:
            return self._process_compound_statement(cursor)
        elif cursor.kind == CursorKind.IF_STMT:
            return self._process_if_statement(cursor)
        elif cursor.kind == CursorKind.FOR_STMT:
            return self._process_for_statement(cursor)
        elif cursor.kind == CursorKind.WHILE_STMT:
            return self._process_while_statement(cursor)
        elif cursor.kind == CursorKind.RETURN_STMT:
            return self._process_return_statement(cursor)
        elif cursor.kind == CursorKind.DECL_STMT:
            return self._process_declaration_statement(cursor)
        elif cursor.kind == CursorKind.CALL_EXPR:
            return self._process_call_expression(cursor)

        # Create a generic statement for other cases
        return CUDAStatement(
            kind=stmt_kind,
            line=cursor.location.line,
            column=cursor.location.column
        )

    def _process_struct_or_class(self, cursor: Cursor) -> CUDANode:
        """Process a struct or class declaration."""
        # Basic processing for struct/class - can be expanded
        node = CUDANode(
            kind=str(cursor.kind),
            spelling=cursor.spelling,
            line=cursor.location.line,
            column=cursor.location.column
        )

        # Process members
        for child in cursor.get_children():
            member = self._process_cursor(child)
            if member:
                node.add_child(member)

        return node

    def _process_typedef(self, cursor: Cursor) -> CUDANode:
        """Process a typedef declaration."""
        return CUDANode(
            kind=str(cursor.kind),
            spelling=cursor.spelling,
            line=cursor.location.line,
            column=cursor.location.column
        )

    def _process_compound_statement(self, cursor: Cursor) -> CUDAStatement:
        """Process a compound statement (block)."""
        stmt = CUDAStatement(
            kind="compound",
            line=cursor.location.line,
            column=cursor.location.column
        )

        for child in cursor.get_children():
            child_stmt = self._process_statement(child)
            if child_stmt:
                stmt.add_child(child_stmt)

        return stmt

    def _process_if_statement(self, cursor: Cursor) -> CUDAStatement:
        """Process an if statement."""
        stmt = CUDAStatement(
            kind="if",
            line=cursor.location.line,
            column=cursor.location.column
        )

        # Process condition, then-branch, and else-branch
        children = list(cursor.get_children())
        if len(children) >= 1:
            stmt.condition = self._process_expression(children[0])

        if len(children) >= 2:
            then_branch = self._process_statement(children[1])
            if then_branch:
                stmt.then_branch = [then_branch]

        if len(children) >= 3:
            else_branch = self._process_statement(children[2])
            if else_branch:
                stmt.else_branch = [else_branch]

        return stmt

    def _process_for_statement(self, cursor: Cursor) -> CUDAStatement:
        """Process a for statement."""
        stmt = CUDAStatement(
            kind="for",
            line=cursor.location.line,
            column=cursor.location.column
        )

        children = list(cursor.get_children())

        # Process initialization, condition, increment, and body
        if len(children) >= 3:
            stmt.init = self._process_expression(children[0])
            stmt.condition = self._process_expression(children[1])
            stmt.increment = self._process_expression(children[2])

        if len(children) >= 4:
            body = self._process_statement(children[3])
            if body:
                stmt.body = [body]

        return stmt

    def _process_while_statement(self, cursor: Cursor) -> CUDAStatement:
        """Process a while statement."""
        stmt = CUDAStatement(
            kind="while",
            line=cursor.location.line,
            column=cursor.location.column
        )

        children = list(cursor.get_children())

        # Process condition and body
        if len(children) >= 1:
            stmt.condition = self._process_expression(children[0])

        if len(children) >= 2:
            body = self._process_statement(children[1])
            if body:
                stmt.body = [body]

        return stmt

    def _process_return_statement(self, cursor: Cursor) -> CUDAStatement:
        """Process a return statement."""
        stmt = CUDAStatement(
            kind="return",
            line=cursor.location.line,
            column=cursor.location.column
        )

        # Process return value if any
        children = list(cursor.get_children())
        if children:
            stmt.expression = self._process_expression(children[0])

        return stmt

    def _process_declaration_statement(self, cursor: Cursor) -> CUDAStatement:
        """Process a declaration statement."""
        stmt = CUDAStatement(
            kind="declaration",
            line=cursor.location.line,
            column=cursor.location.column
        )

        # Process all declarations in this statement
        for child in cursor.get_children():
            var = self._process_variable(child)
            if var:
                stmt.add_child(var)

        return stmt

    def _process_call_expression(self, cursor: Cursor) -> CUDAStatement:
        """Process a function call."""
        stmt = CUDAStatement(
            kind="call",
            line=cursor.location.line,
            column=cursor.location.column
        )

        # Set the function name
        stmt.expression = CUDAExpressionNode(
            spelling=cursor.spelling,
            kind="call_expr",
            line=cursor.location.line,
            column=cursor.location.column
        )

        # Process arguments
        for arg in cursor.get_arguments():
            arg_expr = self._process_expression(arg)
            if arg_expr:
                stmt.expression.add_argument(arg_expr)

        return stmt

    def _process_expression(self, cursor: Cursor) -> CUDAExpressionNode:
        """Process an expression."""
        expr = CUDAExpressionNode(
            spelling=cursor.spelling,
            kind=str(cursor.kind),
            line=cursor.location.line,
            column=cursor.location.column
        )

        # Process different expression types
        if cursor.kind == CursorKind.BINARY_OPERATOR:
            self._process_binary_operator(cursor, expr)
        elif cursor.kind == CursorKind.UNARY_OPERATOR:
            self._process_unary_operator(cursor, expr)
        elif cursor.kind == CursorKind.CALL_EXPR:
            self._process_call_expr(cursor, expr)

        return expr

    def _process_binary_operator(self, cursor: Cursor, expr: CUDAExpressionNode):
        """Process a binary operator expression."""
        children = list(cursor.get_children())
        if len(children) >= 2:
            expr.left = self._process_expression(children[0])
            expr.right = self._process_expression(children[1])

            # Try to determine the operator
            tokens = list(cursor.get_tokens())
            for i, token in enumerate(tokens):
                if i > 0 and token.kind.name == 'PUNCTUATION':
                    expr.operator = token.spelling
                    break

    def _process_unary_operator(self, cursor: Cursor, expr: CUDAExpressionNode):
        """Process a unary operator expression."""
        children = list(cursor.get_children())
        if children:
            expr.operand = self._process_expression(children[0])

            # Try to determine the operator
            tokens = list(cursor.get_tokens())
            if tokens and tokens[0].kind.name == 'PUNCTUATION':
                expr.operator = tokens[0].spelling

    def _process_call_expr(self, cursor: Cursor, expr: CUDAExpressionNode):
        """Process a function call expression."""
        # Get function name
        func_cursor = cursor.referenced
        if func_cursor:
            expr.function = func_cursor.spelling

        # Process arguments
        for arg in cursor.get_arguments():
            arg_expr = self._process_expression(arg)
            if arg_expr:
                expr.add_argument(arg_expr)

    def _process_kernel_attributes(self, cursor: Cursor, kernel: KernelNode):
        """Process CUDA kernel attributes like __launch_bounds__."""
        # Extract launch bounds if present
        for child in cursor.get_children():
            if child.kind == CursorKind.INTEGER_LITERAL:
                # Try to extract the value
                tokens = list(child.get_tokens())
                if tokens:
                    try:
                        value = int(tokens[0].spelling)
                        kernel.max_threads_per_block = value
                    except (ValueError, IndexError):
                        pass

    def _get_type_spelling(self, type_obj) -> str:
        """Get string representation of a type."""
        return type_obj.spelling

    def _get_qualifiers(self, cursor: Cursor) -> List[CUDAQualifier]:
        """Extract qualifiers from a cursor."""
        qualifiers = []

        # Check for const qualifier
        if cursor.type.is_const_qualified():
            qualifiers.append(CUDAQualifier.CONST)

        # Check for CUDA specific qualifiers
        for child in cursor.get_children():
            if child.kind == CursorKind.CUDA_SHARED_ATTR:
                qualifiers.append(CUDAQualifier.SHARED)
            elif child.kind == CursorKind.CUDA_DEVICE_ATTR:
                qualifiers.append(CUDAQualifier.DEVICE)
            elif child.kind == CursorKind.CUDA_CONSTANT_ATTR:
                qualifiers.append(CUDAQualifier.CONST)

        return qualifiers