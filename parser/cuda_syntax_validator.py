# cuda_syntax_validator.py

import re
from typing import List, Dict, Set, Optional, Tuple, Any
from enum import Enum
import clang.cindex
from clang.cindex import CursorKind, TypeKind

from ..utils.error_handler import CudaParseError, raise_cuda_parse_error
from ..utils.logger import get_logger

logger = get_logger(__name__)

class CudaVersion(Enum):
    """Supported CUDA versions"""
    CUDA_8_0 = "8.0"
    CUDA_9_0 = "9.0"
    CUDA_10_0 = "10.0"
    CUDA_11_0 = "11.0"
    CUDA_12_0 = "12.0"

class CudaSyntaxValidator:
    """
    Validates CUDA syntax and ensures compatibility with Metal translation.
    Provides detailed error reporting and suggestions for incompatible features.
    """

    def __init__(self, cuda_version: CudaVersion = CudaVersion.CUDA_11_0):
        self.cuda_version = cuda_version
        self.index = clang.cindex.Index.create()
        self.translation_unit = None

        # Initialize validation rules
        self._init_validation_rules()

        # Tracking state
        self.errors: List[Dict] = []
        self.warnings: List[Dict] = []
        self.unsupported_features: Set[str] = set()

    def _init_validation_rules(self):
        """Initialize validation rules based on CUDA version."""
        self.disallowed_features = {
            # Features not supported in Metal
            'texture1D',
            'texture3D',
            'cudaTextureObject3D',
            '__launch_bounds__',
            'cooperative_groups',
            'dynamic_parallelism',

            # CUDA-specific intrinsics without direct Metal equivalents
            '__ballot_sync',
            '__match_all_sync',
            '__match_any_sync',
            '__activemask',
        }

        self.warning_features = {
            # Features that may need manual optimization
            'atomicAdd',  # Needs special handling in Metal
            'warpSize',   # Different in Metal
            '__syncthreads',  # Different synchronization model
        }

        self.version_specific_features = {
            CudaVersion.CUDA_11_0: {
                'cooperative_groups',
                'cudaLaunchCooperativeKernel',
            }
        }

    def validate_file(self, file_path: str) -> Tuple[bool, List[Dict]]:
        """
        Validate a CUDA source file.

        Args:
            file_path: Path to CUDA source file

        Returns:
            Tuple of (is_valid, list of errors/warnings)
        """
        try:
            self.translation_unit = self.index.parse(
                file_path,
                args=['-x', 'cuda', '--cuda-gpu-arch=sm_70'],
                options=clang.cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD
            )
        except Exception as e:
            raise_cuda_parse_error(f"Failed to parse CUDA file: {str(e)}", filename=file_path)

        # Clear previous state
        self.errors.clear()
        self.warnings.clear()
        self.unsupported_features.clear()

        # Validate translation unit
        self._validate_translation_unit(self.translation_unit.cursor)

        # Check for errors in the translation unit
        for diag in self.translation_unit.diagnostics:
            if diag.severity >= diag.Error:
                self.errors.append({
                    'line': diag.location.line,
                    'column': diag.location.column,
                    'message': diag.spelling,
                    'severity': 'error'
                })
            elif diag.severity == diag.Warning:
                self.warnings.append({
                    'line': diag.location.line,
                    'column': diag.location.column,
                    'message': diag.spelling,
                    'severity': 'warning'
                })

        return len(self.errors) == 0, self.errors + self.warnings

    def _validate_translation_unit(self, cursor: clang.cindex.Cursor):
        """Recursively validate the translation unit."""
        self._validate_node(cursor)
        for child in cursor.get_children():
            self._validate_translation_unit(child)

    def _validate_node(self, node: clang.cindex.Cursor):
        """Validate a single AST node."""
        # Check for disallowed features
        if self._is_disallowed_feature(node):
            self.errors.append({
                'line': node.location.line,
                'column': node.location.column,
                'message': f"Feature '{node.spelling}' is not supported in Metal",
                'severity': 'error',
                'feature': node.spelling
            })
            self.unsupported_features.add(node.spelling)

        # Check for warning features
        if self._is_warning_feature(node):
            self.warnings.append({
                'line': node.location.line,
                'column': node.location.column,
                'message': f"Feature '{node.spelling}' may require manual optimization in Metal",
                'severity': 'warning',
                'feature': node.spelling
            })

        # Validate memory spaces
        if node.kind == CursorKind.VAR_DECL:
            self._validate_memory_space(node)

        # Validate kernel functions
        if self._is_kernel_function(node):
            self._validate_kernel_function(node)

        # Validate atomic operations
        if self._is_atomic_operation(node):
            self._validate_atomic_operation(node)

        # Validate texture operations
        if self._is_texture_operation(node):
            self._validate_texture_operation(node)

    def _validate_memory_space(self, node: clang.cindex.Cursor):
        """Validate memory space declarations."""
        storage_class = node.storage_class

        if storage_class == clang.cindex.StorageClass.CUDA_DEVICE:
            # Validate device memory usage
            pass
        elif storage_class == clang.cindex.StorageClass.CUDA_CONSTANT:
            # Validate constant memory usage
            self._validate_constant_memory(node)
        elif storage_class == clang.cindex.StorageClass.CUDA_SHARED:
            # Validate shared memory usage
            self._validate_shared_memory(node)

    def _validate_kernel_function(self, node: clang.cindex.Cursor):
        """Validate CUDA kernel function."""
        # Check parameter types
        for param in node.get_arguments():
            param_type = param.type
            if not self._is_valid_kernel_parameter_type(param_type):
                self.errors.append({
                    'line': param.location.line,
                    'column': param.location.column,
                    'message': f"Invalid kernel parameter type: {param_type.spelling}",
                    'severity': 'error'
                })

        # Check function attributes
        attrs = node.get_children()
        for attr in attrs:
            if attr.kind == CursorKind.CUDA_GLOBAL_ATTR:
                self._validate_kernel_attributes(attr)

    def _validate_atomic_operation(self, node: clang.cindex.Cursor):
        """Validate atomic operations."""
        # Check if atomic operation is supported in Metal
        op_name = node.spelling
        if not self._is_supported_atomic_operation(op_name):
            self.errors.append({
                'line': node.location.line,
                'column': node.location.column,
                'message': f"Atomic operation '{op_name}' is not supported in Metal",
                'severity': 'error'
            })

        # Check operand types
        for arg in node.get_arguments():
            if not self._is_valid_atomic_operand_type(arg.type):
                self.warnings.append({
                    'line': arg.location.line,
                    'column': arg.location.column,
                    'message': f"Atomic operation on type {arg.type.spelling} may have different behavior in Metal",
                    'severity': 'warning'
                })

    def _validate_texture_operation(self, node: clang.cindex.Cursor):
        """Validate texture operations."""
        # Check texture dimensionality
        tex_type = node.type
        if self._is_unsupported_texture_type(tex_type):
            self.errors.append({
                'line': node.location.line,
                'column': node.location.column,
                'message': f"Texture type {tex_type.spelling} is not supported in Metal",
                'severity': 'error'
            })

        # Check texture access patterns
        for child in node.get_children():
            if child.kind == CursorKind.MEMBER_REF_EXPR:
                self._validate_texture_access(child)

    def _is_disallowed_feature(self, node: clang.cindex.Cursor) -> bool:
        """Check if node represents a disallowed feature."""
        if node.spelling in self.disallowed_features:
            return True

        # Check version-specific features
        if self.cuda_version in self.version_specific_features:
            version_features = self.version_specific_features[self.cuda_version]
            return node.spelling in version_features

        return False

    def _is_warning_feature(self, node: clang.cindex.Cursor) -> bool:
        """Check if node represents a feature that should generate a warning."""
        return node.spelling in self.warning_features

    def _is_kernel_function(self, node: clang.cindex.Cursor) -> bool:
        """Check if node is a CUDA kernel function."""
        return (node.kind == CursorKind.FUNCTION_DECL and
                any(child.kind == CursorKind.CUDA_GLOBAL_ATTR
                    for child in node.get_children()))

    def _is_atomic_operation(self, node: clang.cindex.Cursor) -> bool:
        """Check if node is an atomic operation."""
        return (node.kind == CursorKind.CALL_EXPR and
                node.spelling.startswith('atomic'))

    def _is_texture_operation(self, node: clang.cindex.Cursor) -> bool:
        """Check if node is a texture operation."""
        return (node.kind == CursorKind.CALL_EXPR and
                ('tex' in node.spelling.lower() or
                 'texture' in node.spelling.lower()))

    def _is_valid_kernel_parameter_type(self, type_obj: clang.cindex.Type) -> bool:
        """Check if type is valid for kernel parameters."""
        # Basic types are always valid
        if type_obj.kind in [TypeKind.VOID, TypeKind.BOOL, TypeKind.INT,
                             TypeKind.FLOAT, TypeKind.DOUBLE]:
            return True

        # Pointer types need to be checked
        if type_obj.kind == TypeKind.POINTER:
            pointee = type_obj.get_pointee()
            return self._is_valid_kernel_parameter_type(pointee)

        # Array types need special handling
        if type_obj.kind == TypeKind.CONSTANTARRAY:
            element_type = type_obj.get_array_element_type()
            return self._is_valid_kernel_parameter_type(element_type)

        return False

    def _is_supported_atomic_operation(self, op_name: str) -> bool:
        """Check if atomic operation is supported in Metal."""
        supported_atomics = {
            'atomicAdd',
            'atomicSub',
            'atomicExch',
            'atomicMin',
            'atomicMax',
            'atomicAnd',
            'atomicOr',
            'atomicXor',
        }
        return op_name in supported_atomics

    def _is_valid_atomic_operand_type(self, type_obj: clang.cindex.Type) -> bool:
        """Check if type is valid for atomic operations."""
        valid_types = [
            TypeKind.INT,
            TypeKind.UINT,
            TypeKind.LONG,
            TypeKind.ULONG,
        ]
        return type_obj.kind in valid_types

    def _is_unsupported_texture_type(self, type_obj: clang.cindex.Type) -> bool:
        """Check if texture type is unsupported in Metal."""
        type_spelling = type_obj.spelling.lower()
        return ('texture1d' in type_spelling or
                'texture3d' in type_spelling or
                'cubemap' in type_spelling)

    def _validate_constant_memory(self, node: clang.cindex.Cursor):
        """Validate constant memory usage."""
        # Check size limitations
        if hasattr(node, 'type') and hasattr(node.type, 'get_size'):
            size = node.type.get_size()
            if size > 64 * 1024:  # Metal constant buffer size limit
                self.warnings.append({
                    'line': node.location.line,
                    'column': node.location.column,
                    'message': f"Constant memory size ({size} bytes) exceeds Metal's recommended limit",
                    'severity': 'warning'
                })

    def _validate_shared_memory(self, node: clang.cindex.Cursor):
        """Validate shared memory usage."""
        # Check size limitations
        if hasattr(node, 'type') and hasattr(node.type, 'get_size'):
            size = node.type.get_size()
            if size > 32 * 1024:  # Metal threadgroup memory size limit
                self.errors.append({
                    'line': node.location.line,
                    'column': node.location.column,
                    'message': f"Shared memory size ({size} bytes) exceeds Metal's limit",
                    'severity': 'error'
                })

    def _validate_kernel_attributes(self, attr_node: clang.cindex.Cursor):
        """Validate kernel attributes."""
        # Check for unsupported attributes
        unsupported_attrs = {
            'maxntidx',
            'maxnreg',
            'dynamic_shared_mem_size'
        }

        for child in attr_node.get_children():
            if child.spelling in unsupported_attrs:
                self.warnings.append({
                    'line': child.location.line,
                    'column': child.location.column,
                    'message': f"Kernel attribute '{child.spelling}' is not supported in Metal",
                    'severity': 'warning'
                })

    def _validate_texture_access(self, node: clang.cindex.Cursor):
        """Validate texture access patterns."""
        # Check for unsupported texture operations
        unsupported_ops = {
            'getLod',
            'getGrad',
            'fetch',
        }

        if node.spelling in unsupported_ops:
            self.warnings.append({
                'line': node.location.line,
                'column': node.location.column,
                'message': f"Texture operation '{node.spelling}' may not have direct equivalent in Metal",
                'severity': 'warning'
            })

        # Validate texture coordinates
        for arg in node.get_arguments():
            if not self._is_valid_texture_coordinate(arg):
                self.errors.append({
                    'line': arg.location.line,
                    'column': arg.location.column,
                    'message': f"Invalid texture coordinate type: {arg.type.spelling}",
                    'severity': 'error'
                })

    def _is_valid_texture_coordinate(self, node: clang.cindex.Cursor) -> bool:
        """Check if node represents a valid texture coordinate."""
        valid_types = {
            TypeKind.FLOAT,
            TypeKind.INT,
            TypeKind.UINT
        }
        return node.type.kind in valid_types

    def get_diagnostics(self) -> Dict[str, List[Dict]]:
        """Get all diagnostic messages."""
        return {
            'errors': self.errors,
            'warnings': self.warnings,
            'unsupported_features': list(self.unsupported_features)
        }

    def get_metal_compatibility_report(self) -> Dict[str, Any]:
        """Generate a detailed Metal compatibility report."""
        return {
            'cuda_version': self.cuda_version.value,
            'is_compatible': len(self.errors) == 0,
            'error_count': len(self.errors),
            'warning_count': len(self.warnings),
            'unsupported_features': list(self.unsupported_features),
            'required_changes': self._generate_required_changes(),
            'optimization_suggestions': self._generate_optimization_suggestions()
        }

    def _generate_required_changes(self) -> List[Dict]:
        """Generate list of required changes for Metal compatibility."""
        changes = []

        # Group errors by type
        error_types = {}
        for error in self.errors:
            error_type = error.get('feature', 'other')
            if error_type not in error_types:
                error_types[error_type] = []
            error_types[error_type].append(error)

        # Generate change requirements
        for feature, errors in error_types.items():
            change = {
                'feature': feature,
                'count': len(errors),
                'locations': [{'line': e['line'], 'column': e['column']} for e in errors],
                'suggestion': self._get_change_suggestion(feature)
            }
            changes.append(change)

        return changes

    def _generate_optimization_suggestions(self) -> List[Dict]:
        """Generate optimization suggestions for better Metal performance."""
        suggestions = []

        # Memory access patterns
        if self._has_uncoalesced_memory_access():
            suggestions.append({
                'type': 'memory_access',
                'description': 'Optimize memory access patterns for coalescing',
                'importance': 'high'
            })

        # Thread hierarchy
        if self._has_suboptimal_thread_hierarchy():
            suggestions.append({
                'type': 'thread_hierarchy',
                'description': 'Adjust thread hierarchy for Metal\'s SIMD width',
                'importance': 'medium'
            })

        # Atomic operations
        if self._has_heavy_atomic_usage():
            suggestions.append({
                'type': 'atomic_operations',
                'description': 'Consider alternative algorithms to reduce atomic operations',
                'importance': 'high'
            })

        return suggestions

    def _get_change_suggestion(self, feature: str) -> str:
        """Get suggestion for handling unsupported feature."""
        suggestions = {
            'texture1D': 'Use texture2D with height=1 instead',
            'texture3D': 'Consider restructuring algorithm to use multiple texture2D layers',
            '__launch_bounds__': 'Remove launch bounds and use Metal\'s threadgroup size defaults',
            'cooperative_groups': 'Restructure algorithm to use Metal\'s threading model',
            'dynamic_parallelism': 'Flatten kernel hierarchy or split into multiple passes',
            '__ballot_sync': 'Use Metal\'s simd_vote instead',
            '__match_all_sync': 'Use Metal\'s simd_all instead',
            '__match_any_sync': 'Use Metal\'s simd_any instead',
            '__activemask': 'Use Metal\'s simd_active_threads_mask instead'
        }

        return suggestions.get(feature, 'Requires manual adaptation for Metal')

    def _has_uncoalesced_memory_access(self) -> bool:
        """Check for uncoalesced memory access patterns."""
        # Analyze memory access patterns in the AST
        uncoalesced = False

        def visit(node):
            nonlocal uncoalesced
            if self._is_array_access(node):
                if not self._is_coalesced_access(node):
                    uncoalesced = True
            for child in node.get_children():
                visit(child)

        if self.translation_unit:
            visit(self.translation_unit.cursor)

        return uncoalesced

    def _has_suboptimal_thread_hierarchy(self) -> bool:
        """Check for suboptimal thread hierarchy."""
        for node in self.translation_unit.cursor.walk_preorder():
            if self._is_kernel_function(node):
                dim = self._get_thread_dimensions(node)
                if not self._is_optimal_thread_dim(dim):
                    return True
        return False

    def _has_heavy_atomic_usage(self) -> bool:
        """Check for heavy atomic operation usage."""
        atomic_count = 0
        threshold = 10  # Arbitrary threshold for "heavy" usage

        for node in self.translation_unit.cursor.walk_preorder():
            if self._is_atomic_operation(node):
                atomic_count += 1
                if atomic_count > threshold:
                    return True

        return False

    def _is_array_access(self, node: clang.cindex.Cursor) -> bool:
        """Check if node represents array access."""
        return node.kind == CursorKind.ARRAY_SUBSCRIPT_EXPR

    def _is_coalesced_access(self, node: clang.cindex.Cursor) -> bool:
        """Check if array access is coalesced."""
        # Check if innermost index is thread index
        index = None
        for child in node.get_children():
            if child.kind == CursorKind.INTEGER_LITERAL:
                index = child

        if not index:
            return False

        return self._is_thread_index_based(index)

    def _is_thread_index_based(self, node: clang.cindex.Cursor) -> bool:
        """Check if expression is based on thread index."""
        if node.kind == CursorKind.UNEXPOSED_EXPR:
            for child in node.get_children():
                if 'threadIdx' in child.spelling:
                    return True
        return False

    def _get_thread_dimensions(self, kernel_node: clang.cindex.Cursor) -> Optional[Tuple[int, int, int]]:
        """Extract thread dimensions from kernel launch parameters."""
        for node in kernel_node.walk_preorder():
            if node.spelling == 'blockDim':
                dims = []
                for child in node.get_children():
                    if child.kind == CursorKind.INTEGER_LITERAL:
                        dims.append(child.get_tokens().next().spelling)
                if len(dims) == 3:
                    return tuple(map(int, dims))
        return None

    def _is_optimal_thread_dim(self, dim: Optional[Tuple[int, int, int]]) -> bool:
        """Check if thread dimensions are optimal for Metal."""
        if not dim:
            return False

        x, y, z = dim

        # Check if total threads is within Metal limits
        total_threads = x * y * z
        if total_threads > 1024:  # Metal maximum threads per threadgroup
            return False

        # Check if x dimension is multiple of SIMD width
        if x % 32 != 0:  # Metal SIMD width is 32
            return False

        return True

logger.info("CudaSyntaxValidator initialized for CUDA code validation.")
