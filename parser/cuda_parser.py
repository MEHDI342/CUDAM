import os
import re
import sys
import json
import logging
from typing import List, Dict, Any, Optional, Union, Set, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import clang
import clang.cindex
from clang.cindex import CursorKind, TypeKind, TranslationUnit, AccessSpecifier

from .ast import (
    CudaASTNode, FunctionNode, KernelNode, VariableNode, StructNode,
    EnumNode, TypedefNode, ClassNode, NamespaceNode, TemplateNode,
    CudaKernelLaunchNode, TextureNode, ConstantMemoryNode, SharedMemoryNode,
    CompoundStmtNode, ExpressionNode, DeclRefExprNode, IntegerLiteralNode,
    FloatingLiteralNode, ArraySubscriptNode, BinaryOpNode, UnaryOpNode,
    CallExprNode, MemberExprNode, CastExprNode, InitListExprNode,
    ConditionalOperatorNode, ForStmtNode, WhileStmtNode, DoStmtNode,
    IfStmtNode, SwitchStmtNode, ReturnStmtNode, ContinueStmtNode,
    BreakStmtNode, NullStmtNode
)

from ..utils.error_handler import (
    CudaParseError, CudaTranslationError, CudaTypeError,
    CudaNotSupportedError, CudaWarning
)
from ..utils.logger import get_logger
from ..utils.cuda_builtin_functions import CUDA_BUILTIN_FUNCTIONS
from ..utils.cuda_to_metal_type_mapping import CUDA_TO_METAL_TYPE_MAP
from ..utils.metal_equivalents import METAL_EQUIVALENTS
from ..utils.metal_math_functions import METAL_MATH_FUNCTIONS
from ..utils.metal_optimization_patterns import METAL_OPTIMIZATION_PATTERNS

logger = get_logger(__name__)

class MetalTranslationContext:
    def __init__(self):
        self.buffer_index = 0
        self.texture_index = 0
        self.threadgroup_memory_size = 0
        self.used_metal_features: Set[str] = set()
        self.required_headers: Set[str] = set()
        self.metal_function_declarations: List[str] = []
        self.metal_type_declarations: List[str] = []

class CudaParser:
    """
    Enhanced CUDA Parser with comprehensive Metal translation support.
    This class provides complete parsing and analysis capabilities for CUDA code,
    with robust error handling and optimization strategies.
    """

    def __init__(self, cuda_include_paths: List[str] = None,
                 plugins: Optional[List[Any]] = None,
                 optimization_level: int = 2):
        """
        Initialize the CUDA Parser with enhanced capabilities.

        Args:
            cuda_include_paths: List of paths to CUDA include directories
            plugins: Optional list of parser plugins for extended functionality
            optimization_level: Level of optimization (0-3, higher means more aggressive)
        """
        self.cuda_include_paths = cuda_include_paths or [
            '/usr/local/cuda/include',
            '/usr/local/cuda/samples/common/inc',
            '/opt/cuda/include',
            *self._find_system_cuda_paths()
        ]
        self.index = clang.cindex.Index.create()
        self.translation_unit = None
        self.ast_cache = {}
        self.cuda_builtin_functions = CUDA_BUILTIN_FUNCTIONS
        self.cuda_to_metal_type_map = CUDA_TO_METAL_TYPE_MAP
        self.plugins = plugins or []
        self.metal_equivalents = METAL_EQUIVALENTS
        self.optimization_level = optimization_level
        self.metal_context = MetalTranslationContext()

        # Enhanced configuration
        self.max_thread_group_size = 1024  # Maximum thread group size for Metal
        self.max_total_threads_per_threadgroup = 1024
        self.simd_group_size = 32
        self.max_buffer_size = 1 << 30  # 1GB maximum buffer size

        # Performance optimizations
        self.parallel_parsing = True
        self.cache_enabled = True
        self.aggressive_inlining = optimization_level > 1
        self.enable_metal_fast_math = optimization_level > 0

        # Analysis capabilities
        self.perform_dataflow_analysis = True
        self.perform_alias_analysis = True
        self.enable_advanced_optimizations = optimization_level > 2

        # Initialize libclang
        self._configure_libclang()

    def _configure_libclang(self):
        """Configure libclang with enhanced error handling."""
        try:
            libclang_path = self._find_clang_library()
            clang.cindex.Config.set_library_file(libclang_path)

            # Configure clang features
            self.index.translation_unit_flags = (
                    TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD |
                    TranslationUnit.PARSE_INCOMPLETE |
                    TranslationUnit.PARSE_CACHE_COMPLETION_RESULTS
            )
        except Exception as e:
            logger.error(f"Failed to configure libclang: {str(e)}")
            raise CudaParseError(f"libclang configuration failed: {str(e)}")

    def _find_system_cuda_paths(self) -> List[str]:
        """Find system-wide CUDA installation paths."""
        cuda_paths = []
        possible_locations = [
            '/usr/local/cuda',
            '/opt/cuda',
            '/usr/cuda',
            os.path.expanduser('~/cuda'),
            *os.environ.get('CUDA_PATH', '').split(os.pathsep),
        ]

        for location in possible_locations:
            if os.path.isdir(location):
                include_path = os.path.join(location, 'include')
                if os.path.isdir(include_path):
                    cuda_paths.append(include_path)

        return cuda_paths

    def _find_clang_library(self) -> str:
        """Find and validate libclang library with enhanced path detection."""
        possible_paths = [
            # Linux paths
            '/usr/lib/llvm-10/lib/libclang.so',
            '/usr/lib/llvm-11/lib/libclang.so',
            '/usr/lib/llvm-12/lib/libclang.so',
            '/usr/lib/llvm-13/lib/libclang.so',
            '/usr/lib/llvm-14/lib/libclang.so',
            '/usr/lib/x86_64-linux-gnu/libclang-10.so',
            '/usr/lib/x86_64-linux-gnu/libclang-11.so',
            '/usr/lib/x86_64-linux-gnu/libclang-12.so',
            # macOS paths
            '/usr/local/opt/llvm/lib/libclang.dylib',
            '/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib/libclang.dylib',
            '/Library/Developer/CommandLineTools/usr/lib/libclang.dylib',
            # Windows paths
            'C:/Program Files/LLVM/bin/libclang.dll',
            # Generic paths
            '/usr/local/lib/libclang.so',
            '/usr/lib/libclang.so',
        ]

        # Environment variable override
        if 'LIBCLANG_PATH' in os.environ:
            custom_path = os.environ['LIBCLANG_PATH']
            if os.path.isfile(custom_path):
                return custom_path

        # Search for valid libclang
        for path in possible_paths:
            if os.path.isfile(path):
                try:
                    # Validate the library
                    clang.cindex.Config.set_library_file(path)
                    clang.cindex.Index.create()
                    return path
                except Exception:
                    continue

        raise CudaParseError(
            "libclang library not found. Please install Clang and set the library path.\n"
            "You can set LIBCLANG_PATH environment variable to specify the location."
        )
    def parse_file(self, file_path: str) -> CudaASTNode:
        """
        Parse a CUDA source file with enhanced error handling and caching.

        Args:
            file_path: Path to the CUDA source file

        Returns:
            CudaASTNode: Root node of the parsed AST

        Raises:
            CudaParseError: If parsing fails
            FileNotFoundError: If the file doesn't exist
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CUDA source file not found: {file_path}")

        # Check cache
        if self.cache_enabled and file_path in self.ast_cache:
            cache_entry = self.ast_cache[file_path]
            if os.path.getmtime(file_path) <= cache_entry['timestamp']:
                logger.info(f"Using cached AST for {file_path}")
                return cache_entry['ast']

        try:
            args = self._get_enhanced_clang_args()
            self.translation_unit = self.index.parse(
                file_path,
                args=args,
                options=TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD |
                        TranslationUnit.PARSE_INCOMPLETE |
                        TranslationUnit.PARSE_CACHE_COMPLETION_RESULTS
            )

            # Enhanced error checking
            self._check_diagnostics()

            # Parse with advanced features
            ast = self._enhanced_convert_cursor(self.translation_unit.cursor)

            # Perform additional analysis
            if self.perform_dataflow_analysis:
                self._perform_dataflow_analysis(ast)
            if self.perform_alias_analysis:
                self._perform_alias_analysis(ast)

            # Cache the result
            if self.cache_enabled:
                self.ast_cache[file_path] = {
                    'ast': ast,
                    'timestamp': os.path.getmtime(file_path)
                }

            return ast

        except clang.cindex.TranslationUnitLoadError as e:
            logger.error(f"Error parsing file {file_path}: {str(e)}")
            raise CudaParseError(f"Unable to parse file: {file_path}", details=str(e))
        except Exception as e:
            logger.error(f"Unexpected error parsing {file_path}: {str(e)}")
            raise

    def _get_enhanced_clang_args(self) -> List[str]:
        """Get enhanced clang arguments with comprehensive CUDA support."""
        base_args = [
            '-x', 'cuda',
            '--cuda-gpu-arch=sm_75',
            '-std=c++17',
            '-D__CUDACC__',
            '-D__CUDA_ARCH__=750',
            '-DNDEBUG',
        ]

        cuda_specific_args = [
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
            '-D__CUDA_NO_BFLOAT16_CONVERSIONS__',
            '-D__CUDA_ARCH_LIST__=750',
            '-D__CUDA_PREC_DIV=1',
            '-D__CUDA_PREC_SQRT=1',
        ]

        optimization_args = []
        if self.optimization_level > 0:
            optimization_args.extend([
                '-O2',
                '-ffast-math',
                '-fno-strict-aliasing'
            ])

        include_paths = [f'-I{path}' for path in self.cuda_include_paths]

        return base_args + cuda_specific_args + optimization_args + include_paths

    def _check_diagnostics(self):
        """Enhanced diagnostic checking with detailed error reporting."""
        errors = []
        warnings = []

        for diag in self.translation_unit.diagnostics:
            if diag.severity >= diag.Error:
                errors.append({
                    'severity': diag.severity,
                    'message': diag.spelling,
                    'location': f"{diag.location.file}:{diag.location.line}:{diag.location.column}",
                    'ranges': [
                        (r.start.line, r.start.column, r.end.line, r.end.column)
                        for r in diag.ranges
                    ],
                    'fixits': [f.spelling for f in diag.fixits]
                })
            elif diag.severity == diag.Warning:
                warnings.append({
                    'message': diag.spelling,
                    'location': f"{diag.location.file}:{diag.location.line}:{diag.location.column}"
                })

        # Log warnings
        for warning in warnings:
            logger.warning(f"Clang Warning: {warning['message']} at {warning['location']}")

        # Raise error if there are any
        if errors:
            error_messages = "\n".join([
                f"Error at {error['location']}: {error['message']}"
                for error in errors
            ])
            raise CudaParseError(f"Errors occurred during parsing:\n{error_messages}")

    def _enhanced_convert_cursor(self, cursor: clang.cindex.Cursor) -> CudaASTNode:
        """
        Enhanced cursor conversion with comprehensive CUDA construct handling.

        Args:
            cursor: Clang cursor to convert

        Returns:
            CudaASTNode: Converted AST node
        """
        # Plugin handling
        for plugin in self.plugins:
            result = plugin.handle_cursor(cursor)
            if result:
                return result

        try:
            # Basic node conversion
            node = self._convert_basic_cursor(cursor)
            if node:
                return node

            # Enhanced CUDA-specific handling
            if cursor.kind == CursorKind.CUDA_GLOBAL_ATTR:
                return self._convert_kernel(cursor)
            elif cursor.kind == CursorKind.CUDA_DEVICE_ATTR:
                return self._convert_device_function(cursor)
            elif cursor.kind == CursorKind.CUDA_SHARED_ATTR:
                return self._convert_shared_memory(cursor)
            elif cursor.kind == CursorKind.CUDA_CONSTANT_ATTR:
                return self._convert_constant_memory(cursor)
            elif cursor.kind == CursorKind.CUDA_MANAGED_ATTR:
                return self._convert_managed_memory(cursor)
            elif cursor.kind == CursorKind.CUDA_KERNEL_CALL:
                return self._convert_kernel_launch(cursor)

            # Default handling
            return self._convert_default_cursor(cursor)

        except Exception as e:
            logger.error(f"Error converting cursor {cursor.spelling}: {str(e)}")
            raise CudaParseError(f"Failed to convert cursor: {cursor.spelling}", details=str(e))

    def _convert_basic_cursor(self, cursor: clang.cindex.Cursor) -> Optional[CudaASTNode]:
        """Convert basic C++ constructs to AST nodes."""
        conversion_map = {
            CursorKind.TRANSLATION_UNIT: self._convert_translation_unit,
            CursorKind.NAMESPACE: self._convert_namespace,
            CursorKind.CLASS_DECL: self._convert_class,
            CursorKind.STRUCT_DECL: self._convert_struct,
            CursorKind.ENUM_DECL: self._convert_enum,
            CursorKind.FUNCTION_DECL: self._convert_function,
            CursorKind.VAR_DECL: self._convert_variable,
            CursorKind.FIELD_DECL: self._convert_field,
            CursorKind.TYPEDEF_DECL: self._convert_typedef,
            CursorKind.CXX_METHOD: self._convert_method,
            CursorKind.CONSTRUCTOR: self._convert_constructor,
            CursorKind.DESTRUCTOR: self._convert_destructor,
            CursorKind.COMPOUND_STMT: self._convert_compound_stmt,
            CursorKind.RETURN_STMT: self._convert_return_stmt,
            CursorKind.IF_STMT: self._convert_if_stmt,
            CursorKind.FOR_STMT: self._convert_for_stmt,
            CursorKind.WHILE_STMT: self._convert_while_stmt,
            CursorKind.DO_STMT: self._convert_do_stmt,
            CursorKind.BREAK_STMT: self._convert_break_stmt,
            CursorKind.CONTINUE_STMT: self._convert_continue_stmt,
            CursorKind.CASE_STMT: self._convert_case_stmt,
            CursorKind.SWITCH_STMT: self._convert_switch_stmt,
            CursorKind.BINARY_OPERATOR: self._convert_binary_operator,
            CursorKind.UNARY_OPERATOR: self._convert_unary_operator,
            CursorKind.CALL_EXPR: self._convert_call_expr,
            CursorKind.MEMBER_REF_EXPR: self._convert_member_ref,
            CursorKind.ARRAY_SUBSCRIPT_EXPR: self._convert_array_subscript,
            CursorKind.CONDITIONAL_OPERATOR: self._convert_conditional_operator,
            CursorKind.INIT_LIST_EXPR: self._convert_init_list,
        }

        converter = conversion_map.get(cursor.kind)
        if converter:
            return converter(cursor)
        return None

    def _convert_default_cursor(self, cursor: clang.cindex.Cursor) -> CudaASTNode:
        """Default conversion for unhandled cursor types."""
        children = [self._enhanced_convert_cursor(c) for c in cursor.get_children()]
        return CudaASTNode(
            kind=cursor.kind.name,
            spelling=cursor.spelling,
            type=cursor.type.spelling,
            children=children,
            location=self._get_cursor_location(cursor)
        )

    def _get_cursor_location(self, cursor: clang.cindex.Cursor) -> Dict[str, Any]:
        """Get detailed cursor location information."""
        location = cursor.location
        extent = cursor.extent

        return {
            'file': str(location.file),
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
    def _convert_kernel(self, cursor: clang.cindex.Cursor) -> KernelNode:
        """
        Convert CUDA kernel function to KernelNode with enhanced analysis.

        Args:
            cursor: Clang cursor representing a CUDA kernel

        Returns:
            KernelNode: Node representing the CUDA kernel with analysis metadata
        """
        parameters = [self._convert_variable(arg) for arg in cursor.get_arguments()]
        body = [self._enhanced_convert_cursor(c) for c in cursor.get_children()
                if c.kind != CursorKind.PARM_DECL]

        # Enhanced kernel analysis
        kernel_metadata = self._analyze_kernel(cursor, body)

        return KernelNode(
            name=cursor.spelling,
            parameters=parameters,
            body=body,
            attributes=self._get_function_attributes(cursor),
            launch_config=self._extract_launch_config(cursor),
            metadata=kernel_metadata,
            location=self._get_cursor_location(cursor)
        )

    def _analyze_kernel(self, cursor: clang.cindex.Cursor, body: List[CudaASTNode]) -> Dict[str, Any]:
        """Perform comprehensive kernel analysis."""
        return {
            'memory_access_patterns': self._analyze_memory_patterns(body),
            'thread_hierarchy': self._analyze_thread_hierarchy(body),
            'synchronization_points': self._find_sync_points(body),
            'arithmetic_intensity': self._calculate_arithmetic_intensity(body),
            'register_pressure': self._estimate_register_pressure(body),
            'shared_memory_usage': self._analyze_shared_memory_usage(body),
            'data_dependencies': self._analyze_data_dependencies(body),
            'control_flow_complexity': self._analyze_control_flow(body),
            'optimization_opportunities': self._identify_optimization_opportunities(body),
            'metal_compatibility': self._check_metal_compatibility(body)
        }

    def _analyze_memory_patterns(self, nodes: List[CudaASTNode]) -> Dict[str, Any]:
        """Analyze memory access patterns for optimization."""
        patterns = {
            'coalesced_accesses': [],
            'uncoalesced_accesses': [],
            'shared_memory_accesses': [],
            'texture_accesses': [],
            'constant_memory_accesses': [],
            'global_memory_accesses': [],
            'bank_conflicts': [],
            'stride_patterns': {},
        }

        def analyze_node(node: CudaASTNode):
            if isinstance(node, ArraySubscriptNode):
                access_info = self._classify_memory_access(node)
                if access_info['type'] == 'coalesced':
                    patterns['coalesced_accesses'].append(access_info)
                else:
                    patterns['uncoalesced_accesses'].append(access_info)

                if access_info.get('stride_pattern'):
                    patterns['stride_patterns'][node.spelling] = access_info['stride_pattern']

            elif isinstance(node, CallExprNode):
                if self._is_texture_operation(node):
                    patterns['texture_accesses'].append(self._analyze_texture_access(node))
                elif self._is_shared_memory_operation(node):
                    patterns['shared_memory_accesses'].append(
                        self._analyze_shared_memory_access(node)
                    )

            for child in node.children:
                analyze_node(child)

        for node in nodes:
            analyze_node(node)

        return patterns

    def _analyze_thread_hierarchy(self, nodes: List[CudaASTNode]) -> Dict[str, Any]:
        """Analyze thread hierarchy and synchronization patterns."""
        hierarchy = {
            'block_size': self._extract_block_size(nodes),
            'grid_size': self._extract_grid_size(nodes),
            'thread_coarsening_factor': self._calculate_thread_coarsening(nodes),
            'sync_patterns': self._analyze_sync_patterns(nodes),
            'warp_level_ops': self._find_warp_operations(nodes),
            'thread_divergence': self._analyze_thread_divergence(nodes),
        }

        # Optimize for Metal
        hierarchy['metal_threadgroup_size'] = self._optimize_threadgroup_size(
            hierarchy['block_size']
        )
        hierarchy['metal_grid_size'] = self._optimize_grid_size(
            hierarchy['grid_size'],
            hierarchy['metal_threadgroup_size']
        )

        return hierarchy

    def _find_sync_points(self, nodes: List[CudaASTNode]) -> List[Dict[str, Any]]:
        """Identify and analyze synchronization points."""
        sync_points = []

        def analyze_sync(node: CudaASTNode, context: Dict[str, Any]):
            if isinstance(node, CallExprNode):
                if node.spelling == '__syncthreads':
                    sync_points.append({
                        'type': 'block_sync',
                        'location': node.location,
                        'context': context.copy(),
                        'scope': self._analyze_sync_scope(node),
                        'dependencies': self._analyze_sync_dependencies(node),
                    })
                elif node.spelling == '__threadfence':
                    sync_points.append({
                        'type': 'device_fence',
                        'location': node.location,
                        'context': context.copy(),
                        'scope': 'device',
                    })
                elif node.spelling == '__threadfence_block':
                    sync_points.append({
                        'type': 'block_fence',
                        'location': node.location,
                        'context': context.copy(),
                        'scope': 'block',
                    })

            # Recursive analysis with context
            current_context = context.copy()
            if isinstance(node, (IfStmtNode, ForStmtNode, WhileStmtNode)):
                current_context['control_structure'] = node.__class__.__name__
                current_context['condition'] = str(node.condition)

            for child in node.children:
                analyze_sync(child, current_context)

        analyze_sync(nodes, {})
        return sync_points

    def _analyze_sync_scope(self, node: CallExprNode) -> Dict[str, Any]:
        """Analyze the scope and impact of a synchronization point."""
        return {
            'affected_variables': self._find_affected_variables(node),
            'critical_section': self._identify_critical_section(node),
            'barrier_type': self._determine_barrier_type(node),
            'optimization_potential': self._evaluate_sync_optimization(node),
        }

    def _calculate_arithmetic_intensity(self, nodes: List[CudaASTNode]) -> float:
        """Calculate arithmetic intensity (operations per memory access)."""
        operations = 0
        memory_accesses = 0

        def count_operations(node: CudaASTNode):
            nonlocal operations, memory_accesses

            if isinstance(node, (BinaryOpNode, UnaryOpNode)):
                operations += 1
            elif isinstance(node, ArraySubscriptNode):
                memory_accesses += 1
            elif isinstance(node, CallExprNode):
                if self._is_math_operation(node):
                    operations += self._get_operation_cost(node)
                elif self._is_memory_operation(node):
                    memory_accesses += 1

            for child in node.children:
                count_operations(child)

        for node in nodes:
            count_operations(node)

        return operations / max(memory_accesses, 1)

    def _estimate_register_pressure(self, nodes: List[CudaASTNode]) -> Dict[str, Any]:
        """Estimate register pressure and provide optimization suggestions."""
        registers = {
            'local_vars': set(),
            'temp_vars': set(),
            'loop_vars': set(),
            'max_live_vars': 0,
            'spill_estimate': 0,
            'optimization_suggestions': []
        }

        def analyze_registers(node: CudaASTNode, scope: Dict[str, Set[str]]):
            if isinstance(node, VariableNode):
                if node.storage_class == 'auto':
                    registers['local_vars'].add(node.name)
                    scope['current'].add(node.name)
            elif isinstance(node, ForStmtNode):
                registers['loop_vars'].add(node.init.name)

            live_vars = len(scope['current'])
            registers['max_live_vars'] = max(registers['max_live_vars'], live_vars)

            if live_vars > 128:  # Metal maximum register count
                registers['spill_estimate'] += 1
                registers['optimization_suggestions'].append({
                    'type': 'register_pressure',
                    'location': node.location,
                    'suggestion': 'Consider splitting kernel or reducing local variables'
                })

            for child in node.children:
                analyze_registers(child, scope)

        analyze_registers(nodes, {'current': set()})
        return registers

    def _analyze_data_dependencies(self, nodes: List[CudaASTNode]) -> Dict[str, Any]:
        """Analyze data dependencies for parallelization opportunities."""
        dependencies = {
            'flow_dependencies': [],
            'anti_dependencies': [],
            'output_dependencies': [],
            'parallel_regions': [],
            'critical_paths': [],
            'vectorization_opportunities': []
        }

        def analyze_deps(node: CudaASTNode, context: Dict[str, Any]):
            if isinstance(node, ArraySubscriptNode):
                deps = self._analyze_array_dependencies(node)
                dependencies['flow_dependencies'].extend(deps['flow'])
                dependencies['anti_dependencies'].extend(deps['anti'])
                dependencies['output_dependencies'].extend(deps['output'])

            elif isinstance(node, ForStmtNode):
                if self._is_parallelizable_loop(node):
                    dependencies['parallel_regions'].append({
                        'node': node,
                        'type': 'loop',
                        'optimization': 'vectorization'
                    })

            for child in node.children:
                analyze_deps(child, context)

        analyze_deps(nodes, {})
        return dependencies
    def _analyze_control_flow(self, nodes: List[CudaASTNode]) -> Dict[str, Any]:
        """
        Analyze control flow patterns for Metal optimization opportunities.
        """
        control_flow = {
            'branch_density': 0,
            'loop_nesting_depth': 0,
            'divergent_branches': [],
            'uniform_branches': [],
            'loop_trip_counts': {},
            'vectorizable_loops': [],
            'unrollable_loops': [],
            'critical_paths': [],
            'metal_optimizations': []
        }

        def analyze_node(node: CudaASTNode, context: Dict[str, Any]):
            if isinstance(node, IfStmtNode):
                branch_info = self._analyze_branch(node)
                if branch_info['is_divergent']:
                    control_flow['divergent_branches'].append(branch_info)
                else:
                    control_flow['uniform_branches'].append(branch_info)

                # Metal-specific optimizations
                if branch_info['is_divergent']:
                    control_flow['metal_optimizations'].append({
                        'type': 'branch_optimization',
                        'location': node.location,
                        'suggestion': 'Consider using select() for better Metal performance'
                    })

            elif isinstance(node, ForStmtNode):
                loop_info = self._analyze_loop(node)
                control_flow['loop_trip_counts'][node] = loop_info['trip_count']

                if loop_info['is_vectorizable']:
                    control_flow['vectorizable_loops'].append(loop_info)
                if loop_info['is_unrollable']:
                    control_flow['unrollable_loops'].append(loop_info)

                # Metal-specific loop optimizations
                if loop_info['trip_count'] <= 8:
                    control_flow['metal_optimizations'].append({
                        'type': 'loop_unroll',
                        'location': node.location,
                        'suggestion': 'Unroll small loop for Metal performance'
                    })

            # Update metrics
            control_flow['branch_density'] = self._calculate_branch_density(nodes)
            control_flow['loop_nesting_depth'] = max(
                control_flow['loop_nesting_depth'],
                context.get('nesting_depth', 0)
            )

            # Recursive analysis
            new_context = context.copy()
            if isinstance(node, (ForStmtNode, WhileStmtNode)):
                new_context['nesting_depth'] = context.get('nesting_depth', 0) + 1

            for child in node.children:
                analyze_node(child, new_context)

        analyze_node(nodes, {'nesting_depth': 0})
        return control_flow

    def _analyze_branch(self, node: IfStmtNode) -> Dict[str, Any]:
        """Analyze branch characteristics for Metal optimization."""
        condition_vars = self._extract_condition_variables(node.condition)

        return {
            'is_divergent': any(self._is_thread_dependent(var) for var in condition_vars),
            'condition_complexity': self._calculate_condition_complexity(node.condition),
            'branch_balance': self._calculate_branch_balance(node),
            'condition_variables': condition_vars,
            'optimization_potential': self._evaluate_branch_optimization(node),
            'metal_transforms': self._get_metal_branch_transforms(node)
        }

    def _analyze_loop(self, node: ForStmtNode) -> Dict[str, Any]:
        """Analyze loop characteristics for Metal optimization."""
        return {
            'trip_count': self._calculate_trip_count(node),
            'is_vectorizable': self._check_vectorizable(node),
            'is_unrollable': self._check_unrollable(node),
            'iteration_dependencies': self._analyze_iteration_dependencies(node),
            'memory_access_pattern': self._analyze_loop_memory_pattern(node),
            'metal_parallel_mapping': self._get_metal_parallel_mapping(node),
            'optimization_strategy': self._determine_loop_optimization_strategy(node)
        }

    def _get_metal_parallel_mapping(self, node: ForStmtNode) -> Dict[str, Any]:
        """Determine how to map loop iterations to Metal's parallel execution model."""
        trip_count = self._calculate_trip_count(node)
        dependencies = self._analyze_iteration_dependencies(node)

        if not dependencies['has_dependencies']:
            if trip_count <= self.max_thread_group_size:
                return {
                    'strategy': 'threadgroup',
                    'size': trip_count,
                    'code_template': self._generate_metal_threadgroup_code(node)
                }
            else:
                return {
                    'strategy': 'grid',
                    'size': (trip_count + self.max_thread_group_size - 1)
                            // self.max_thread_group_size,
                    'code_template': self._generate_metal_grid_code(node)
                }
        else:
            return {
                'strategy': 'sequential',
                'optimization': self._get_sequential_optimization_strategy(node),
                'code_template': self._generate_metal_sequential_code(node)
            }

    def _translate_to_metal_kernel(self, node: KernelNode) -> str:
        """Translate CUDA kernel to Metal kernel with optimizations."""
        metal_code = []

        # Generate kernel signature
        params = self._translate_kernel_parameters(node.parameters)
        metal_code.append(f"kernel void {node.name}({params})")
        metal_code.append("{")

        # Generate thread indexing code
        metal_code.extend(self._generate_metal_thread_indexing())

        # Translate kernel body with optimizations
        body_code = self._translate_kernel_body(node.body)
        metal_code.extend([f"    {line}" for line in body_code])

        metal_code.append("}")

        return "\n".join(metal_code)

    def _translate_kernel_parameters(self, params: List[VariableNode]) -> str:
        """Translate CUDA kernel parameters to Metal parameters."""
        metal_params = []

        for idx, param in enumerate(params):
            metal_type = self._cuda_type_to_metal(param.data_type)

            if param.is_pointer():
                if param.is_readonly():
                    qualifier = "const device"
                else:
                    qualifier = "device"
                metal_params.append(
                    f"{qualifier} {metal_type}* {param.name} [[buffer({idx})]]"
                )
            else:
                metal_params.append(
                    f"constant {metal_type}& {param.name} [[buffer({idx})]]"
                )

        return ", ".join(metal_params)

    def _generate_metal_thread_indexing(self) -> List[str]:
        """Generate Metal-specific thread indexing code."""
        return [
            "    const uint3 thread_position_in_grid [[thread_position_in_grid]];",
            "    const uint3 threads_per_grid [[threads_per_grid]];",
            "    const uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]];",
            "    const uint3 threads_per_threadgroup [[threads_per_threadgroup]];",
            "    const uint3 threadgroup_position [[threadgroup_position_in_grid]];",
            "",
            "    const uint global_id = thread_position_in_grid.x +",
            "                          thread_position_in_grid.y * threads_per_grid.x +",
            "                          thread_position_in_grid.z * threads_per_grid.x * threads_per_grid.y;",
            ""
        ]

    def _translate_kernel_body(self, body: List[CudaASTNode]) -> List[str]:
        """Translate CUDA kernel body to Metal with optimizations."""
        metal_code = []

        # Apply optimizations
        optimized_body = self._optimize_for_metal(body)

        # Translate each node
        for node in optimized_body:
            metal_code.extend(self._translate_node_to_metal(node))

        return metal_code

    def _optimize_for_metal(self, nodes: List[CudaASTNode]) -> List[CudaASTNode]:
        """Apply Metal-specific optimizations to the AST."""
        optimizations = [
            self._optimize_memory_access_patterns,
            self._optimize_thread_synchronization,
            self._optimize_arithmetic_operations,
            self._optimize_control_flow,
            self._optimize_function_calls
        ]

        optimized_nodes = nodes
        for optimization in optimizations:
            optimized_nodes = optimization(optimized_nodes)

        return optimized_nodes

    def _translate_node_to_metal(self, node: CudaASTNode) -> List[str]:
        """Translate a single AST node to Metal code."""
        if isinstance(node, ArraySubscriptNode):
            return self._translate_array_access(node)
        elif isinstance(node, CallExprNode):
            return self._translate_function_call(node)
        elif isinstance(node, IfStmtNode):
            return self._translate_if_statement(node)
        elif isinstance(node, ForStmtNode):
            return self._translate_for_loop(node)
        elif isinstance(node, BinaryOpNode):
            return self._translate_binary_operation(node)
        elif isinstance(node, UnaryOpNode):
            return self._translate_unary_operation(node)
        # Add more node type translations...

        return [f"// Unsupported node type: {node.__class__.__name__}"]
    def _translate_array_access(self, node: ArraySubscriptNode) -> List[str]:
        """
        Translate CUDA array access to optimized Metal code.
        """
        base = self._translate_expression(node.array)
        index = self._translate_expression(node.index)

        # Check if this is a texture access
        if self._is_texture_access(node):
            return self._generate_texture_access(base, index)

        # Check if this is a shared memory access
        if self._is_shared_memory_access(node):
            return self._generate_threadgroup_access(base, index)

        # Optimize global memory access
        if self._is_global_memory_access(node):
            return self._generate_optimized_global_access(base, index)

        return [f"{base}[{index}]"]

    def _generate_texture_access(self, base: str, index: str) -> List[str]:
        """Generate optimized Metal texture access code."""
        texture_info = self.metal_context.texture_mappings.get(base, {})
        coord_type = "float2" if texture_info.get("dimensions", 2) == 2 else "float3"

        return [
            f"constexpr sampler textureSampler(coord::pixel, address::clamp_to_edge, filter::linear);",
            f"const {coord_type} textureCoord = {self._convert_index_to_texture_coord(index)};",
            f"{base}.sample(textureSampler, textureCoord)"
        ]

    def _generate_threadgroup_access(self, base: str, index: str) -> List[str]:
        """Generate optimized Metal threadgroup memory access."""
        # Check for bank conflicts
        if self._has_bank_conflicts(base, index):
            return self._generate_bank_conflict_free_access(base, index)

        return [f"threadgroup_memory[{index}]"]

    def _generate_optimized_global_access(self, base: str, index: str) -> List[str]:
        """Generate coalesced Metal global memory access."""
        stride_pattern = self._analyze_stride_pattern(index)

        if stride_pattern == "coalesced":
            return [f"{base}[{index}]"]
        else:
            return self._generate_optimized_strided_access(base, index)

    def _translate_function_call(self, node: CallExprNode) -> List[str]:
        """Translate CUDA function calls to Metal equivalents."""
        # Handle built-in CUDA functions
        if node.name in self.cuda_builtin_functions:
            return self._translate_builtin_function(node)

        # Handle math functions
        if node.name in METAL_MATH_FUNCTIONS:
            return self._translate_math_function(node)

        # Handle atomic operations
        if self._is_atomic_operation(node):
            return self._translate_atomic_operation(node)

        # Handle texture operations
        if self._is_texture_operation(node):
            return self._translate_texture_operation(node)

        # Handle regular function calls
        return self._translate_regular_function_call(node)

    def _translate_builtin_function(self, node: CallExprNode) -> List[str]:
        """Translate CUDA built-in functions to Metal equivalents."""
        metal_equivalent = self.metal_equivalents[node.name]
        translated_args = [self._translate_expression(arg) for arg in node.arguments]

        if callable(metal_equivalent):
            return metal_equivalent(*translated_args)

        return [f"{metal_equivalent}({', '.join(translated_args)})"]

    def _translate_atomic_operation(self, node: CallExprNode) -> List[str]:
        """Translate CUDA atomic operations to Metal atomics."""
        atomic_map = {
            'atomicAdd': 'atomic_fetch_add_explicit',
            'atomicSub': 'atomic_fetch_sub_explicit',
            'atomicMax': 'atomic_fetch_max_explicit',
            'atomicMin': 'atomic_fetch_min_explicit',
            'atomicAnd': 'atomic_fetch_and_explicit',
            'atomicOr': 'atomic_fetch_or_explicit',
            'atomicXor': 'atomic_fetch_xor_explicit',
            'atomicCAS': 'atomic_compare_exchange_weak_explicit'
        }

        if node.name not in atomic_map:
            raise CudaTranslationError(f"Unsupported atomic operation: {node.name}")

        metal_atomic = atomic_map[node.name]
        translated_args = [self._translate_expression(arg) for arg in node.arguments]
        memory_order = 'memory_order_relaxed'

        return [f"{metal_atomic}({', '.join(translated_args)}, {memory_order})"]

    def _translate_control_flow(self, node: CudaASTNode) -> List[str]:
        """Translate control flow structures to Metal."""
        if isinstance(node, IfStmtNode):
            return self._translate_if_statement(node)
        elif isinstance(node, ForStmtNode):
            return self._translate_for_loop(node)
        elif isinstance(node, WhileStmtNode):
            return self._translate_while_loop(node)
        elif isinstance(node, DoStmtNode):
            return self._translate_do_loop(node)
        elif isinstance(node, SwitchStmtNode):
            return self._translate_switch_statement(node)
        else:
            return self._translate_generic_statement(node)

    def _translate_if_statement(self, node: IfStmtNode) -> List[str]:
        """Translate if statements with Metal optimizations."""
        condition = self._translate_expression(node.condition)

        # Check if we can use select() instead
        if self._can_use_select(node):
            return self._generate_select_statement(node)

        metal_code = [f"if ({condition}) {{"]
        metal_code.extend(self._translate_body(node.then_branch))

        if node.else_branch:
            metal_code.append("} else {")
            metal_code.extend(self._translate_body(node.else_branch))

        metal_code.append("}")
        return metal_code

    def _translate_for_loop(self, node: ForStmtNode) -> List[str]:
        """Translate for loops with Metal optimizations."""
        # Check for optimization opportunities
        if self._can_unroll_loop(node):
            return self._generate_unrolled_loop(node)

        if self._can_vectorize_loop(node):
            return self._generate_vectorized_loop(node)

        # Regular translation
        init = self._translate_expression(node.init)
        condition = self._translate_expression(node.condition)
        increment = self._translate_expression(node.increment)

        metal_code = [f"for ({init}; {condition}; {increment}) {{"]
        metal_code.extend(self._translate_body(node.body))
        metal_code.append("}")
        return metal_code

    def _generate_unrolled_loop(self, node: ForStmtNode) -> List[str]:
        """Generate unrolled loop code for Metal."""
        trip_count = self._calculate_trip_count(node)
        metal_code = []

        for i in range(trip_count):
            loop_body = self._translate_body(node.body)
            replaced_body = self._replace_loop_variable(loop_body, node.init.name, str(i))
            metal_code.extend(replaced_body)

        return metal_code

    def _generate_vectorized_loop(self, node: ForStmtNode) -> List[str]:
        """Generate vectorized loop code for Metal."""
        vector_size = self._determine_vector_size(node)
        metal_code = []

        # Generate vector declarations
        metal_code.extend(self._generate_vector_declarations(node, vector_size))

        # Generate vectorized computation
        metal_code.extend(self._generate_vector_computation(node, vector_size))

        # Generate cleanup code for remaining iterations
        metal_code.extend(self._generate_vector_cleanup(node, vector_size))

        return metal_code

    def _translate_synchronization(self, node: CallExprNode) -> List[str]:
        """Translate CUDA synchronization primitives to Metal."""
        sync_map = {
            '__syncthreads': 'threadgroup_barrier(mem_flags::mem_threadgroup)',
            '__threadfence': 'threadgroup_barrier(mem_flags::mem_device)',
            '__threadfence_block': 'threadgroup_barrier(mem_flags::mem_threadgroup)',
            '__syncwarp': 'simdgroup_barrier(mem_flags::mem_none)'
        }

        if node.name not in sync_map:
            raise CudaTranslationError(f"Unsupported synchronization primitive: {node.name}")

        return [sync_map[node.name] + ";"]
    def _optimize_memory_access_patterns(self, nodes: List[CudaASTNode]) -> List[CudaASTNode]:
        """
        Optimize memory access patterns for Metal performance.
        """
        optimized_nodes = []

        for node in nodes:
            if isinstance(node, ArraySubscriptNode):
                optimized_nodes.append(self._optimize_array_access(node))
            elif isinstance(node, CallExprNode):
                if self._is_memory_operation(node):
                    optimized_nodes.append(self._optimize_memory_operation(node))
                else:
                    optimized_nodes.append(self._optimize_node_recursively(node))
            else:
                optimized_nodes.append(self._optimize_node_recursively(node))

        return optimized_nodes

    def _optimize_array_access(self, node: ArraySubscriptNode) -> CudaASTNode:
        """Optimize array access patterns for Metal."""
        access_pattern = self._analyze_access_pattern(node)

        if access_pattern['type'] == 'strided':
            return self._optimize_strided_access(node, access_pattern)
        elif access_pattern['type'] == 'gather':
            return self._optimize_gather_access(node, access_pattern)
        elif access_pattern['type'] == 'scatter':
            return self._optimize_scatter_access(node, access_pattern)

        # If no specific optimization applies, try to coalesce the access
        return self._coalesce_memory_access(node)

    def _optimize_strided_access(self, node: ArraySubscriptNode, pattern: Dict[str, Any]) -> CudaASTNode:
        """Optimize strided memory access patterns."""
        stride = pattern['stride']

        if stride == 1:
            # Already coalesced
            return node

        if self._is_power_of_two(stride):
            # Use bit manipulation optimization
            return self._generate_optimized_stride_access(node, stride)

        # Generate vectorized access if possible
        if self._can_vectorize_access(node, stride):
            return self._generate_vectorized_access(node, stride)

        return node

    def _optimize_memory_operation(self, node: CallExprNode) -> CudaASTNode:
        """Optimize Metal memory operations."""
        if self._is_texture_operation(node):
            return self._optimize_texture_operation(node)
        elif self._is_shared_memory_operation(node):
            return self._optimize_threadgroup_memory_operation(node)
        elif self._is_atomic_operation(node):
            return self._optimize_atomic_operation(node)

        return node

    def _optimize_texture_operation(self, node: CallExprNode) -> CudaASTNode:
        """Optimize texture operations for Metal."""
        # Add Metal-specific texture optimizations
        texture_info = self._analyze_texture_usage(node)

        if texture_info['access_pattern'] == 'sequential':
            return self._generate_optimized_texture_access(node)
        elif texture_info['access_pattern'] == 'random':
            return self._generate_cached_texture_access(node)

        return node

    def _optimize_threadgroup_memory_operation(self, node: CallExprNode) -> CudaASTNode:
        """Optimize threadgroup memory operations."""
        # Check for bank conflicts
        if self._has_bank_conflicts(node):
            return self._resolve_bank_conflicts(node)

        # Optimize access pattern
        access_pattern = self._analyze_threadgroup_access_pattern(node)
        if access_pattern['type'] == 'broadcast':
            return self._optimize_broadcast_access(node)
        elif access_pattern['type'] == 'reduction':
            return self._optimize_reduction_access(node)

        return node

    def _optimize_atomic_operation(self, node: CallExprNode) -> CudaASTNode:
        """Optimize atomic operations for Metal."""
        operation_type = self._get_atomic_operation_type(node)

        if operation_type == 'increment':
            return self._optimize_atomic_increment(node)
        elif operation_type == 'reduction':
            return self._optimize_atomic_reduction(node)

        return node

    def _optimize_control_flow(self, nodes: List[CudaASTNode]) -> List[CudaASTNode]:
        """Optimize control flow for Metal performance."""
        optimized_nodes = []

        for node in nodes:
            if isinstance(node, IfStmtNode):
                optimized_nodes.append(self._optimize_conditional(node))
            elif isinstance(node, ForStmtNode):
                optimized_nodes.append(self._optimize_loop(node))
            elif isinstance(node, WhileStmtNode):
                optimized_nodes.append(self._optimize_while_loop(node))
            else:
                optimized_nodes.append(self._optimize_node_recursively(node))

        return optimized_nodes

    def _optimize_conditional(self, node: IfStmtNode) -> CudaASTNode:
        """Optimize conditional statements for Metal."""
        # Check if we can convert to select()
        if self._can_use_select(node):
            return self._convert_to_select(node)

        # Check for thread divergence
        if self._is_divergent_branch(node):
            return self._optimize_divergent_branch(node)

        # Optimize condition evaluation
        optimized_condition = self._optimize_condition(node.condition)
        node.condition = optimized_condition

        # Recursively optimize branches
        node.then_branch = self._optimize_control_flow(node.then_branch)
        if node.else_branch:
            node.else_branch = self._optimize_control_flow(node.else_branch)

        return node

    def _optimize_loop(self, node: ForStmtNode) -> CudaASTNode:
        """Optimize loops for Metal performance."""
        # Check for loop unrolling opportunity
        if self._should_unroll_loop(node):
            return self._unroll_loop(node)

        # Check for vectorization opportunity
        if self._can_vectorize_loop(node):
            return self._vectorize_loop(node)

        # Check for parallel reduction
        if self._is_reduction_loop(node):
            return self._optimize_reduction_loop(node)

        # General loop optimizations
        optimized_init = self._optimize_node_recursively(node.init)
        optimized_condition = self._optimize_node_recursively(node.condition)
        optimized_increment = self._optimize_node_recursively(node.increment)
        optimized_body = self._optimize_control_flow(node.body)

        return ForStmtNode(
            init=optimized_init,
            condition=optimized_condition,
            increment=optimized_increment,
            body=optimized_body
        )

    def _optimize_function_calls(self, nodes: List[CudaASTNode]) -> List[CudaASTNode]:
        """Optimize function calls for Metal."""
        optimized_nodes = []

        for node in nodes:
            if isinstance(node, CallExprNode):
                # Handle special CUDA functions
                if node.name in self.cuda_builtin_functions:
                    optimized_nodes.append(self._optimize_builtin_function(node))
                # Handle math functions
                elif self._is_math_function(node):
                    optimized_nodes.append(self._optimize_math_function(node))
                # Handle custom functions
                else:
                    optimized_nodes.append(self._optimize_custom_function(node))
            else:
                optimized_nodes.append(self._optimize_node_recursively(node))

        return optimized_nodes

    def _optimize_builtin_function(self, node: CallExprNode) -> CudaASTNode:
        """Optimize CUDA built-in function calls for Metal."""
        if node.name in METAL_OPTIMIZATION_PATTERNS:
            return self._apply_optimization_pattern(node)

        # Optimize function arguments
        optimized_args = [self._optimize_node_recursively(arg) for arg in node.arguments]
        node.arguments = optimized_args

        return node

    def _optimize_math_function(self, node: CallExprNode) -> CudaASTNode:
        """Optimize math function calls for Metal."""
        # Use fast math where applicable
        if self.enable_metal_fast_math:
            if node.name in METAL_MATH_FUNCTIONS:
                return self._convert_to_fast_math(node)

        # Optimize common math patterns
        optimized = self._optimize_math_pattern(node)
        if optimized:
            return optimized

        return node

    def _optimize_node_recursively(self, node: CudaASTNode) -> CudaASTNode:
        """Recursively optimize a node and its children."""
        if isinstance(node, (list, tuple)):
            return [self._optimize_node_recursively(child) for child in node]

        # Optimize current node
        optimized_node = self._apply_node_specific_optimizations(node)

        # Recursively optimize children
        if hasattr(optimized_node, 'children'):
            optimized_node.children = [
                self._optimize_node_recursively(child)
                for child in optimized_node.children
            ]

        return optimized_node
    def _apply_node_specific_optimizations(self, node: CudaASTNode) -> CudaASTNode:
        """
        Apply specific optimizations based on node type and context.
        """
        optimization_map = {
            BinaryOpNode: self._optimize_binary_operation,
            UnaryOpNode: self._optimize_unary_operation,
            CallExprNode: self._optimize_function_call,
            ArraySubscriptNode: self._optimize_array_access,
            MemberExprNode: self._optimize_member_access,
            CastExprNode: self._optimize_type_cast,
            ConditionalOperatorNode: self._optimize_conditional_operator
        }

        optimizer = optimization_map.get(type(node))
        if optimizer:
            return optimizer(node)
        return node

    def _optimize_binary_operation(self, node: BinaryOpNode) -> CudaASTNode:
        """Optimize binary operations for Metal."""
        # Check for vector operations
        if self._is_vector_operation(node):
            return self._vectorize_binary_operation(node)

        # Check for constant folding
        if self._can_constant_fold(node):
            return self._fold_constants(node)

        # Check for strength reduction
        if self._can_reduce_strength(node):
            return self._reduce_strength(node)

        # Optimize operands
        node.left = self._optimize_node_recursively(node.left)
        node.right = self._optimize_node_recursively(node.right)

        return node

    def _vectorize_binary_operation(self, node: BinaryOpNode) -> CudaASTNode:
        """Convert binary operation to vectorized Metal operation."""
        vector_info = self._analyze_vector_operation(node)

        if vector_info['vector_width'] == 4:
            return self._create_float4_operation(node)
        elif vector_info['vector_width'] == 2:
            return self._create_float2_operation(node)

        return node

    def _optimize_metal_specific_features(self, nodes: List[CudaASTNode]) -> List[CudaASTNode]:
        """Optimize for Metal-specific features and capabilities."""
        optimizations = [
            self._optimize_simd_group_operations,
            self._optimize_threadgroup_memory_layout,
            self._optimize_texture_sampling,
            self._optimize_buffer_access_patterns,
            self._optimize_compute_pipeline_state,
            self._optimize_argument_buffer_usage
        ]

        result = nodes
        for optimization in optimizations:
            result = optimization(result)
        return result

    def _optimize_simd_group_operations(self, nodes: List[CudaASTNode]) -> List[CudaASTNode]:
        """Optimize operations to utilize Metal's SIMD groups."""
        optimized = []

        for node in nodes:
            if self._is_reducible_operation(node):
                optimized.append(self._convert_to_simd_group_reduction(node))
            elif self._is_broadcast_operation(node):
                optimized.append(self._convert_to_simd_group_broadcast(node))
            elif self._is_shuffle_operation(node):
                optimized.append(self._convert_to_simd_group_shuffle(node))
            else:
                optimized.append(self._optimize_node_recursively(node))

        return optimized

    def _convert_to_simd_group_reduction(self, node: CallExprNode) -> CudaASTNode:
        """Convert reduction operations to use Metal's SIMD group functions."""
        operation_type = self._get_reduction_type(node)

        metal_simd_op = {
            'sum': 'simd_sum',
            'product': 'simd_product',
            'min': 'simd_min',
            'max': 'simd_max',
            'and': 'simd_and',
            'or': 'simd_or',
            'xor': 'simd_xor'
        }.get(operation_type)

        if metal_simd_op:
            return self._create_simd_reduction(node, metal_simd_op)

        return node

    def _optimize_threadgroup_memory_layout(self, nodes: List[CudaASTNode]) -> List[CudaASTNode]:
        """Optimize threadgroup memory layout for Metal."""
        # Analyze threadgroup memory usage
        memory_usage = self._analyze_threadgroup_memory_usage(nodes)

        # Optimize layout based on access patterns
        if memory_usage['has_bank_conflicts']:
            return self._resolve_threadgroup_memory_conflicts(nodes)

        # Optimize for spatial locality
        if memory_usage['needs_padding']:
            return self._add_threadgroup_memory_padding(nodes)

        return nodes

    def _optimize_texture_sampling(self, nodes: List[CudaASTNode]) -> List[CudaASTNode]:
        """Optimize texture sampling operations for Metal."""
        optimized = []

        for node in nodes:
            if isinstance(node, CallExprNode) and self._is_texture_operation(node):
                texture_info = self._analyze_texture_usage(node)

                if texture_info['can_use_linear_sampling']:
                    optimized.append(self._convert_to_linear_sampling(node))
                elif texture_info['can_use_immediate_mode']:
                    optimized.append(self._convert_to_immediate_mode(node))
                else:
                    optimized.append(node)
            else:
                optimized.append(self._optimize_node_recursively(node))

        return optimized

    def _optimize_buffer_access_patterns(self, nodes: List[CudaASTNode]) -> List[CudaASTNode]:
        """Optimize buffer access patterns for Metal."""
        # Analyze buffer access patterns
        access_patterns = self._analyze_buffer_access_patterns(nodes)

        optimized = []
        for node in nodes:
            if isinstance(node, ArraySubscriptNode):
                pattern = access_patterns.get(node.array.name)
                if pattern:
                    if pattern['type'] == 'sequential':
                        optimized.append(self._optimize_sequential_access(node))
                    elif pattern['type'] == 'strided':
                        optimized.append(self._optimize_strided_access(node))
                    elif pattern['type'] == 'random':
                        optimized.append(self._optimize_random_access(node))
                    else:
                        optimized.append(node)
            else:
                optimized.append(self._optimize_node_recursively(node))

        return optimized

    def _create_metal_compute_pipeline(self, kernel_node: KernelNode) -> Dict[str, Any]:
        """Create Metal compute pipeline configuration."""
        return {
            'function_name': kernel_node.name,
            'thread_execution_width': self._calculate_execution_width(kernel_node),
            'max_total_threads_per_threadgroup': self._calculate_max_threads(kernel_node),
            'threadgroup_size_is_multiple_of_thread_execution_width': True,
            'buffer_layouts': self._generate_buffer_layouts(kernel_node),
            'texture_layouts': self._generate_texture_layouts(kernel_node),
            'argument_buffer_layouts': self._generate_argument_buffer_layouts(kernel_node),
            'optimization_hints': self._generate_optimization_hints(kernel_node)
        }

    def _calculate_execution_width(self, kernel_node: KernelNode) -> int:
        """Calculate optimal execution width for Metal."""
        analysis = self._analyze_kernel_characteristics(kernel_node)

        if analysis['vector_width'] == 4:
            return 32  # Optimal for float4 operations
        elif analysis['memory_coalescing']:
            return 32  # Optimal for coalesced memory access
        elif analysis['divergent_branching']:
            return 16  # Better for divergent code
        else:
            return 32  # Default SIMD width

    def _generate_buffer_layouts(self, kernel_node: KernelNode) -> List[Dict[str, Any]]:
        """Generate optimal buffer layouts for Metal."""
        layouts = []

        for param in kernel_node.parameters:
            if self._is_buffer_parameter(param):
                layouts.append({
                    'name': param.name,
                    'index': len(layouts),
                    'type': self._get_metal_buffer_type(param),
                    'access': self._get_buffer_access_type(param),
                    'alignment': self._calculate_buffer_alignment(param),
                    'offset': self._calculate_buffer_offset(param)
                })

        return layouts

    def _generate_optimization_hints(self, kernel_node: KernelNode) -> Dict[str, Any]:
        """Generate optimization hints for Metal compiler."""
        analysis = self._analyze_kernel_characteristics(kernel_node)

        return {
            'preferred_simd_width': self._calculate_execution_width(kernel_node),
            'memory_access_patterns': analysis['memory_patterns'],
            'arithmetic_intensity': analysis['arithmetic_intensity'],
            'branch_divergence': analysis['branch_divergence'],
            'resource_usage': {
                'threadgroup_memory': analysis['threadgroup_memory_size'],
                'registers_per_thread': analysis['registers_per_thread'],
                'texture_usage': analysis['texture_usage']
            },
            'optimization_flags': self._generate_optimization_flags(analysis)
        }
    def _generate_metal_code(self, ast: CudaASTNode) -> str:
        """
        Generate optimized Metal code from the AST.
        """
        metal_code = []

        # Add necessary Metal includes and types
        metal_code.extend(self._generate_metal_headers())

        # Add type definitions and constants
        metal_code.extend(self._generate_metal_types())
        metal_code.extend(self._generate_metal_constants())

        # Generate the actual kernel code
        metal_code.extend(self._generate_kernel_implementations(ast))

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
        if self.metal_context.required_headers:
            headers.extend(self.metal_context.required_headers)
            headers.append("")

        return headers

    def _generate_metal_types(self) -> List[str]:
        """Generate Metal-specific type definitions."""
        type_definitions = []

        # Generate structure definitions
        for struct in self._collect_struct_definitions():
            type_definitions.extend(self._generate_metal_struct(struct))
            type_definitions.append("")

        # Generate custom type aliases
        type_definitions.extend(self._generate_type_aliases())
        type_definitions.append("")

        return type_definitions

    def _generate_metal_constants(self) -> List[str]:
        """Generate Metal constant definitions."""
        constants = []

        # Generate constant buffer definitions
        for const in self._collect_constant_definitions():
            constants.extend(self._generate_constant_buffer(const))

        # Generate compile-time constants
        constants.extend(self._generate_compile_time_constants())

        return constants

    def _generate_kernel_implementations(self, ast: CudaASTNode) -> List[str]:
        """Generate Metal kernel implementations."""
        implementations = []

        # Process each kernel
        for kernel in self._collect_kernels(ast):
            # Generate kernel metadata
            implementations.extend(self._generate_kernel_metadata(kernel))

            # Generate the actual kernel implementation
            implementations.extend(self._generate_metal_kernel(kernel))
            implementations.append("")

            # Generate helper functions for this kernel
            implementations.extend(self._generate_kernel_helpers(kernel))
            implementations.append("")

        return implementations

    def _generate_metal_kernel(self, kernel: KernelNode) -> List[str]:
        """Generate a Metal kernel implementation."""
        metal_code = []

        # Generate kernel signature
        metal_code.append(self._generate_kernel_signature(kernel))
        metal_code.append("{")

        # Generate thread indexing code
        metal_code.extend(self._generate_thread_indexing_code(kernel))

        # Generate local variable declarations
        metal_code.extend(self._generate_local_declarations(kernel))

        # Generate kernel body
        body_code = self._generate_kernel_body(kernel)
        metal_code.extend([f"    {line}" for line in body_code])

        metal_code.append("}")
        return metal_code

    def _generate_kernel_metadata(self, kernel: KernelNode) -> List[str]:
        """Generate Metal kernel metadata and attributes."""
        metadata = []

        # Generate kernel attributes
        metadata.extend([
            f"// Kernel: {kernel.name}",
            "// Attributes:",
            f"//   - Thread Execution Width: {self._calculate_execution_width(kernel)}",
            f"//   - Max Threads Per Threadgroup: {self._calculate_max_threads(kernel)}",
            f"//   - Threadgroup Memory Size: {self._calculate_threadgroup_memory_size(kernel)} bytes",
            "//   - Buffer Bindings:",
        ])

        # Add buffer binding information
        for idx, param in enumerate(kernel.parameters):
            metadata.append(f"//     [{idx}] {param.name}: {self._get_metal_type(param.data_type)}")

        metadata.append("")
        return metadata

    def _generate_thread_indexing_code(self, kernel: KernelNode) -> List[str]:
        """Generate optimized thread indexing code for Metal."""
        indexing_code = [
            "    // Thread and threadgroup indexing",
            "    const uint3 thread_position_in_grid [[thread_position_in_grid]];",
            "    const uint3 threadgroup_position [[threadgroup_position_in_grid]];",
            "    const uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]];",
            "    const uint thread_index = thread_position_in_grid.x +",
            "                             thread_position_in_grid.y * gridDim.x +",
            "                             thread_position_in_grid.z * gridDim.x * gridDim.y;",
            ""
        ]

        # Add SIMD group indexing if needed
        if self._needs_simd_group_indexing(kernel):
            indexing_code.extend([
                "    const uint simd_lane_id = thread_position_in_threadgroup.x & 0x1F;",
                "    const uint simd_group_id = thread_position_in_threadgroup.x >> 5;",
                ""
            ])

        return indexing_code

    def _generate_kernel_body(self, kernel: KernelNode) -> List[str]:
        """Generate optimized kernel body code."""
        body_code = []

        # Generate thread bounds check if needed
        if self._needs_bounds_check(kernel):
            body_code.extend(self._generate_bounds_check(kernel))

        # Generate main computation
        for node in kernel.body:
            body_code.extend(self._generate_node_code(node))

        return body_code

    def _generate_node_code(self, node: CudaASTNode) -> List[str]:
        """Generate Metal code for a specific AST node."""
        generators = {
            ArraySubscriptNode: self._generate_array_access_code,
            BinaryOpNode: self._generate_binary_operation_code,
            CallExprNode: self._generate_function_call_code,
            IfStmtNode: self._generate_if_statement_code,
            ForStmtNode: self._generate_for_loop_code,
            WhileStmtNode: self._generate_while_loop_code,
            ReturnStmtNode: self._generate_return_statement_code,
            VariableNode: self._generate_variable_code,
            CompoundStmtNode: self._generate_compound_statement_code
        }

        generator = generators.get(type(node), self._generate_default_node_code)
        return generator(node)

    def _generate_function_call_code(self, node: CallExprNode) -> List[str]:
        """Generate Metal code for function calls."""
        # Handle special CUDA functions
        if node.name in self.cuda_builtin_functions:
            return self._generate_builtin_function_code(node)

        # Handle math functions
        if self._is_math_function(node):
            return self._generate_math_function_code(node)

        # Handle atomic operations
        if self._is_atomic_operation(node):
            return self._generate_atomic_operation_code(node)

        # Handle texture operations
        if self._is_texture_operation(node):
            return self._generate_texture_operation_code(node)

        # Handle regular function calls
        return self._generate_regular_function_call_code(node)

    def _generate_builtin_function_code(self, node: CallExprNode) -> List[str]:
        """Generate Metal code for CUDA built-in functions."""
        # Get Metal equivalent
        metal_function = self.metal_equivalents.get(node.name)
        if not metal_function:
            raise CudaTranslationError(f"Unsupported CUDA built-in function: {node.name}")

        # Generate argument code
        args = [self._generate_expression_code(arg) for arg in node.arguments]

        # Generate the function call
        if callable(metal_function):
            return metal_function(*args)
        else:
            return [f"{metal_function}({', '.join(args)});"]
    def _generate_atomic_operation_code(self, node: CallExprNode) -> List[str]:
        """
        Generate Metal code for atomic operations with advanced optimization.
        """
        atomic_map = {
            'atomicAdd': ('atomic_fetch_add_explicit', 'memory_order_relaxed'),
            'atomicSub': ('atomic_fetch_sub_explicit', 'memory_order_relaxed'),
            'atomicMin': ('atomic_fetch_min_explicit', 'memory_order_relaxed'),
            'atomicMax': ('atomic_fetch_max_explicit', 'memory_order_relaxed'),
            'atomicInc': ('atomic_fetch_add_explicit', 'memory_order_relaxed'),
            'atomicDec': ('atomic_fetch_sub_explicit', 'memory_order_relaxed'),
            'atomicCAS': ('atomic_compare_exchange_weak_explicit', 'memory_order_relaxed')
        }

        if node.name not in atomic_map:
            raise CudaTranslationError(f"Unsupported atomic operation: {node.name}")

        metal_func, memory_order = atomic_map[node.name]
        args = [self._generate_expression_code(arg) for arg in node.arguments]

        # Handle special cases
        if node.name == 'atomicCAS':
            return [
                f"{metal_func}({args[0]}, {args[1]}, {args[2]}, "
                f"{memory_order}, {memory_order});"
            ]
        else:
            return [f"{metal_func}({', '.join(args)}, {memory_order});"]

    def _generate_texture_operation_code(self, node: CallExprNode) -> List[str]:
        """Generate optimized Metal texture operations."""
        texture_info = self._analyze_texture_operation(node)

        # Generate texture sampler configuration
        sampler_config = self._generate_sampler_configuration(texture_info)

        # Generate texture coordinates
        coord_code = self._generate_texture_coordinates(node.arguments, texture_info)

        # Generate the actual texture operation
        if texture_info['operation_type'] == 'sample':
            return self._generate_texture_sample(node, sampler_config, coord_code)
        elif texture_info['operation_type'] == 'write':
            return self._generate_texture_write(node, coord_code)
        elif texture_info['operation_type'] == 'atomic':
            return self._generate_texture_atomic(node, coord_code)
        else:
            raise CudaTranslationError(f"Unsupported texture operation: {texture_info['operation_type']}")

    def _generate_sampler_configuration(self, texture_info: Dict[str, Any]) -> str:
        """Generate Metal texture sampler configuration."""
        config_parts = []

        # Add sampling mode
        if texture_info.get('linear_filtering', False):
            config_parts.append('filter::linear')
        else:
            config_parts.append('filter::nearest')

        # Add address mode
        address_mode = texture_info.get('address_mode', 'clamp')
        config_parts.append(f'address::{address_mode}_to_edge')

        # Add coordinate space
        if texture_info.get('normalized_coords', False):
            config_parts.append('coord::normalized')
        else:
            config_parts.append('coord::pixel')

        return f"sampler({', '.join(config_parts)})"

    def _generate_math_function_code(self, node: CallExprNode) -> List[str]:
        """Generate optimized Metal math function code."""
        # Check if we can use fast math
        if self.enable_metal_fast_math and node.name in METAL_MATH_FUNCTIONS:
            return self._generate_fast_math_code(node)

        # Generate optimized math operations
        math_map = {
            'pow': self._generate_optimized_pow,
            'exp': self._generate_optimized_exp,
            'log': self._generate_optimized_log,
            'sqrt': self._generate_optimized_sqrt,
            'sin': self._generate_optimized_sin,
            'cos': self._generate_optimized_cos,
            'tan': self._generate_optimized_tan
        }

        if node.name in math_map:
            return math_map[node.name](node)

        # Fall back to standard math functions
        return self._generate_standard_math_code(node)

    def _generate_fast_math_code(self, node: CallExprNode) -> List[str]:
        """Generate fast math operations for Metal."""
        fast_math_map = {
            'sin': 'fast::sin',
            'cos': 'fast::cos',
            'exp': 'fast::exp',
            'exp2': 'fast::exp2',
            'log': 'fast::log',
            'log2': 'fast::log2',
            'pow': 'fast::pow',
            'rsqrt': 'fast::rsqrt',
            'sqrt': 'fast::sqrt',
            'fma': 'fast::fma'
        }

        metal_func = fast_math_map[node.name]
        args = [self._generate_expression_code(arg) for arg in node.arguments]
        return [f"{metal_func}({', '.join(args)});"]

    def _generate_optimized_pow(self, node: CallExprNode) -> List[str]:
        """Generate optimized power function code."""
        base = self._generate_expression_code(node.arguments[0])
        exponent = node.arguments[1]

        # Handle special cases for integer exponents
        if isinstance(exponent, IntegerLiteralNode):
            exp_value = int(exponent.value)

            if exp_value == 0:
                return ["1.0;"]
            elif exp_value == 1:
                return [f"{base};"]
            elif exp_value == 2:
                return [f"({base} * {base});"]
            elif exp_value == 3:
                return [f"({base} * {base} * {base});"]
            elif exp_value == 4:
                return [
                    f"float _temp = {base};",
                    "_temp = _temp * _temp;",
                    "_temp * _temp;"
                ]

        # Use fast::pow for other cases when fast math is enabled
        if self.enable_metal_fast_math:
            return [f"fast::pow({base}, {self._generate_expression_code(exponent)});"]

        return [f"pow({base}, {self._generate_expression_code(exponent)});"]

    def _generate_vector_operation_code(self, node: BinaryOpNode) -> List[str]:
        """Generate optimized vector operation code."""
        left = self._generate_expression_code(node.left)
        right = self._generate_expression_code(node.right)

        # Determine vector width
        vector_width = self._get_vector_width(node)

        if vector_width == 4:
            return self._generate_float4_operation(node.operator, left, right)
        elif vector_width == 2:
            return self._generate_float2_operation(node.operator, left, right)
        else:
            return [f"{left} {node.operator} {right};"]

    def _generate_float4_operation(self, operator: str, left: str, right: str) -> List[str]:
        """Generate optimized float4 vector operations."""
        if operator in {'+', '-', '*', '/'}:
            return [f"{left} {operator} {right};"]
        elif operator == '*=':
            return [f"{left} = {left} * {right};"]
        elif operator == '+=':
            return [f"{left} = {left} + {right};"]
        elif operator == '-=':
            return [f"{left} = {left} - {right};"]
        else:
            # Handle component-wise operations
            return [
                f"float4({left}.x {operator} {right}.x, "
                f"{left}.y {operator} {right}.y, "
                f"{left}.z {operator} {right}.z, "
                f"{left}.w {operator} {right}.w);"
            ]

    def _generate_control_flow_code(self, node: CudaASTNode) -> List[str]:
        """Generate optimized control flow code."""
        if isinstance(node, IfStmtNode):
            return self._generate_if_statement_code(node)
        elif isinstance(node, ForStmtNode):
            return self._generate_for_loop_code(node)
        elif isinstance(node, WhileStmtNode):
            return self._generate_while_loop_code(node)
        elif isinstance(node, SwitchStmtNode):
            return self._generate_switch_statement_code(node)
        elif isinstance(node, DoStmtNode):
            return self._generate_do_loop_code(node)
        else:
            return self._generate_default_control_flow(node)

    def _generate_if_statement_code(self, node: IfStmtNode) -> List[str]:
        """Generate optimized if statement code."""
        # Check if we can use select()
        if self._can_use_select(node):
            return self._generate_select_statement(node)

        condition = self._generate_expression_code(node.condition)
        code = [f"if ({condition}) {{"]

        # Generate then branch
        then_code = self._generate_block_code(node.then_branch)
        code.extend(f"    {line}" for line in then_code)

        # Generate else branch if it exists
        if node.else_branch:
            code.append("} else {")
            else_code = self._generate_block_code(node.else_branch)
            code.extend(f"    {line}" for line in else_code)

        code.append("}")
        return code

    def _generate_select_statement(self, node: IfStmtNode) -> List[str]:
        """Generate Metal select statement for simple conditionals."""
        condition = self._generate_expression_code(node.condition)
        then_expr = self._generate_expression_code(node.then_branch.children[0])
        else_expr = self._generate_expression_code(node.else_branch.children[0])

        return [f"select({else_expr}, {then_expr}, {condition});"]
    def _generate_performance_optimized_code(self, nodes: List[CudaASTNode]) -> List[str]:
        """
        Generate highly optimized Metal code with advanced performance considerations.
        """
        optimized_code = []

        # Pre-optimization analysis
        analysis = self._analyze_optimization_opportunities(nodes)

        # Apply optimization strategies based on analysis
        if analysis['vectorization_possible']:
            nodes = self._apply_vectorization(nodes)
        if analysis['loop_fusion_possible']:
            nodes = self._apply_loop_fusion(nodes)
        if analysis['memory_coalescing_needed']:
            nodes = self._apply_memory_coalescing(nodes)

        # Generate code for each optimized node
        for node in nodes:
            optimized_code.extend(self._generate_optimized_node_code(node))

        return optimized_code

    def _analyze_optimization_opportunities(self, nodes: List[CudaASTNode]) -> Dict[str, Any]:
        """Analyze code for optimization opportunities."""
        return {
            'vectorization_possible': self._check_vectorization_opportunities(nodes),
            'loop_fusion_possible': self._check_loop_fusion_opportunities(nodes),
            'memory_coalescing_needed': self._check_memory_coalescing_needs(nodes),
            'simd_utilization': self._analyze_simd_utilization(nodes),
            'thread_divergence': self._analyze_thread_divergence(nodes),
            'memory_access_patterns': self._analyze_memory_patterns(nodes),
            'compute_intensity': self._calculate_compute_intensity(nodes),
            'register_pressure': self._estimate_register_pressure(nodes),
            'shared_memory_usage': self._analyze_shared_memory_usage(nodes)
        }

    def _analyze_optimization_opportunities(self, nodes: List[CudaASTNode]) -> Dict[str, Any]:
        """Analyze code for optimization opportunities."""
        return {
            'vectorization_possible': self._check_vectorization_opportunities(nodes),
            'loop_fusion_possible': self._check_loop_fusion_opportunities(nodes),
            'memory_coalescing_needed': self._check_memory_coalescing_needs(nodes),
            'simd_utilization': self._analyze_simd_utilization(nodes),
            'thread_divergence': self._analyze_thread_divergence(nodes),
            'memory_access_patterns': self._analyze_memory_patterns(nodes),
            'compute_intensity': self._calculate_compute_intensity(nodes),
            'register_pressure': self._estimate_register_pressure(nodes),
            'shared_memory_usage': self._analyze_shared_memory_usage(nodes)
        }

    def _finalize_metal_code(self, code: List[str]) -> str:
        """
        Finalize Metal code with necessary boilerplate and optimizations.
        """
        final_code = []

        # Add header section
        final_code.extend(self._generate_metal_headers())

        # Add custom type definitions
        final_code.extend(self._generate_custom_types())

        # Add constant definitions
        final_code.extend(self._generate_constants())

        # Add the main code
        final_code.extend(code)

        # Add helper functions
        final_code.extend(self._generate_helper_functions())

        # Perform final optimizations
        optimized_code = self._perform_final_optimizations("\n".join(final_code))

        return optimized_code

    def _perform_final_optimizations(self, code: str) -> str:
        """
        Perform final pass optimizations on the generated Metal code.
        """
        # Remove unnecessary brackets
        code = self._optimize_brackets(code)

        # Optimize variable declarations
        code = self._optimize_variable_declarations(code)

        # Optimize memory barriers
        code = self._optimize_memory_barriers(code)

        # Remove redundant operations
        code = self._remove_redundant_operations(code)

        # Optimize register usage
        code = self._optimize_register_usage(code)

        return code

    def _optimize_register_usage(self, code: str) -> str:
        """Optimize register usage in the generated code."""
        lines = code.split('\n')
        register_map: Dict[str, int] = {}
        optimized_lines = []

        for line in lines:
            # Analyze register usage
            used_registers = self._analyze_line_register_usage(line)

            # Apply register optimization
            if used_registers:
                line = self._optimize_line_registers(line, register_map, used_registers)

            optimized_lines.append(line)

        return '\n'.join(optimized_lines)

    def _generate_simd_optimized_code(self, node: CudaASTNode) -> List[str]:
        """Generate SIMD-optimized Metal code."""
        simd_code = []

        if self._can_use_simd_group(node):
            simd_code.extend(self._generate_simd_group_code(node))
        elif self._can_vectorize(node):
            simd_code.extend(self._generate_vector_code(node))
        else:
            simd_code.extend(self._generate_scalar_code(node))

        return simd_code

    def _generate_memory_optimized_code(self, nodes: List[CudaASTNode]) -> List[str]:
        """Generate memory-optimized Metal code."""
        optimized_code = []

        # Analyze memory access patterns
        access_patterns = self._analyze_memory_access_patterns(nodes)

        for node in nodes:
            if self._is_memory_operation(node):
                optimized_code.extend(
                    self._generate_optimized_memory_operation(node, access_patterns)
                )
            else:
                optimized_code.extend(self._generate_node_code(node))

        return optimized_code

    def _generate_optimized_memory_operation(
            self,
            node: CudaASTNode,
            patterns: Dict[str, Any]
    ) -> List[str]:
        """Generate optimized memory operation code."""
        if self._is_coalesced_access(node, patterns):
            return self._generate_coalesced_access(node)
        elif self._is_cached_access(node, patterns):
            return self._generate_cached_access(node)
        elif self._is_broadcast_access(node, patterns):
            return self._generate_broadcast_access(node)
        else:
            return self._generate_default_memory_access(node)

    def _generate_threadgroup_optimized_code(self, node: CudaASTNode) -> List[str]:
        """Generate threadgroup-optimized Metal code."""
        threadgroup_code = []

        # Analyze threadgroup usage
        threadgroup_info = self._analyze_threadgroup_usage(node)

        # Generate declarations
        threadgroup_code.extend(
            self._generate_threadgroup_declarations(threadgroup_info)
        )

        # Generate synchronization
        if threadgroup_info['needs_synchronization']:
            threadgroup_code.extend(
                self._generate_threadgroup_synchronization(threadgroup_info)
            )

        # Generate main code
        threadgroup_code.extend(
            self._generate_threadgroup_computation(node, threadgroup_info)
        )

        return threadgroup_code

    def _optimize_kernel_launch(self, node: KernelNode) -> Dict[str, Any]:
        """Optimize kernel launch configuration for Metal."""
        return {
            'threadgroup_size': self._calculate_optimal_threadgroup_size(node),
            'grid_size': self._calculate_optimal_grid_size(node),
            'shared_memory_size': self._calculate_shared_memory_size(node),
            'barrier_optimization': self._optimize_barriers(node),
            'memory_layout': self._optimize_memory_layout(node),
            'register_allocation': self._optimize_register_allocation(node)
        }

    def _generate_final_kernel(self,
                               kernel: KernelNode,
                               optimizations: Dict[str, Any]) -> str:
        """Generate the final optimized Metal kernel."""
        # Start with kernel signature
        kernel_code = [
            self._generate_kernel_signature(kernel, optimizations),
            "{"
        ]

        # Add threadgroup memory declarations
        if optimizations['shared_memory_size'] > 0:
            kernel_code.extend(
                self._generate_threadgroup_memory_declarations(optimizations)
            )

        # Add thread indexing
        kernel_code.extend(self._generate_thread_indexing())

        # Add main computation
        kernel_code.extend(
            self._generate_optimized_computation(kernel, optimizations)
        )

        kernel_code.append("}")
        return "\n".join(kernel_code)

    def finalize(self) -> None:
        """
        Perform final cleanup and optimization steps.
        """
        # Clear any temporary data
        self.metal_context.clear_temporary_data()

        # Validate generated code
        self._validate_generated_code()

        # Release resources
        self._cleanup_resources()

    def _validate_generated_code(self) -> None:
        """Validate the generated Metal code."""
        for filename, code in self.metal_context.generated_code.items():
            try:
                self._validate_metal_syntax(code)
                self._validate_resource_usage(code)
                self._validate_performance_characteristics(code)
            except CudaTranslationError as e:
                logger.error(f"Validation failed for {filename}: {str(e)}")
                raise

    def _cleanup_resources(self) -> None:
        """Cleanup temporary resources."""
        self.ast_cache.clear()
        self.translation_unit = None

        # Clear context
        self.metal_context.cleanup()