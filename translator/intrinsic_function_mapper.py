
from typing import Dict, Optional, List, Tuple, Union, Set
from dataclasses import dataclass
from enum import Enum
import logging

from ..utils.error_handler import CudaTranslationError
from ..utils.logger import get_logger

logger = get_logger(__name__)

class IntrinsicType(Enum):
    MATH = "math"
    ATOMIC = "atomic"
    SYNC = "sync"
    MEMORY = "memory"
    THREAD = "thread"
    WARP = "warp"
    SPECIAL = "special"

@dataclass
class IntrinsicFunction:
    """Represents a CUDA intrinsic function with its Metal equivalent."""
    cuda_name: str
    metal_name: str
    return_type: str
    arg_types: List[str]
    type: IntrinsicType
    needs_wrapper: bool = False
    has_metal_equivalent: bool = True
    requires_memory_order: bool = False
    requires_scope: bool = False
    is_simd_function: bool = False
    vectorizable: bool = False
    custom_translation: Optional[str] = None

class IntrinsicFunctionMapper:
    """Maps CUDA intrinsic functions to their Metal equivalents."""

    def __init__(self):
        self.intrinsics: Dict[str, IntrinsicFunction] = self._init_intrinsics()
        self.used_intrinsics: Set[str] = set()
        self.required_headers: Set[str] = set()

    def _init_intrinsics(self) -> Dict[str, IntrinsicFunction]:
        """Initialize all supported intrinsic functions."""
        return {
            # Math intrinsics
            "__sinf": IntrinsicFunction(
                cuda_name="__sinf",
                metal_name="metal::fast::sin",
                return_type="float",
                arg_types=["float"],
                type=IntrinsicType.MATH,
                vectorizable=True
            ),
            "__cosf": IntrinsicFunction(
                cuda_name="__cosf",
                metal_name="metal::fast::cos",
                return_type="float",
                arg_types=["float"],
                type=IntrinsicType.MATH,
                vectorizable=True
            ),
            # ... other intrinsic definitions ...
        }

    def map_intrinsic(self, node: dict) -> str:
        """Map CUDA intrinsic function call to Metal equivalent."""
        try:
            func_name = node.get('function', {}).get('name')
            if not func_name:
                raise CudaTranslationError(f"Invalid intrinsic function call: {node}")

            if func_name not in self.intrinsics:
                raise CudaTranslationError(f"Unknown intrinsic function: {func_name}")

            intrinsic = self.intrinsics[func_name]
            self.used_intrinsics.add(func_name)

            # Handle custom translations
            if intrinsic.custom_translation:
                return intrinsic.custom_translation

            # Generate Metal function call
            args = self._translate_arguments(node.get('arguments', []), intrinsic)
            metal_call = f"{intrinsic.metal_name}({', '.join(args)})"

            # Add memory order if required
            if intrinsic.requires_memory_order:
                metal_call += ", memory_order_relaxed"

            # Add scope if required
            if intrinsic.requires_scope:
                metal_call += "(mem_flags::mem_threadgroup)"

            return metal_call

        except Exception as e:
            logger.error(f"Error mapping intrinsic function: {str(e)}")
            raise CudaTranslationError(f"Failed to map intrinsic function: {str(e)}")

    def _translate_arguments(self, args: List[dict], intrinsic: IntrinsicFunction) -> List[str]:
        """Translate function arguments to Metal."""
        if len(args) != len(intrinsic.arg_types):
            raise CudaTranslationError(
                f"Wrong number of arguments for {intrinsic.cuda_name}: "
                f"expected {len(intrinsic.arg_types)}, got {len(args)}"
            )

        translated_args = []
        for arg, expected_type in zip(args, intrinsic.arg_types):
            arg_str = self._translate_argument(arg, expected_type)
            translated_args.append(arg_str)

        return translated_args

    def _translate_argument(self, arg: dict, expected_type: str) -> str:
        """Translate single argument with type checking."""
        if 'value' in arg:
            return str(arg['value'])
        elif 'name' in arg:
            return arg['name']
        return str(arg)

    def get_required_headers(self) -> Set[str]:
        """Get required Metal headers based on used intrinsics."""
        headers = set()
        for intrinsic_name in self.used_intrinsics:
            intrinsic = self.intrinsics[intrinsic_name]
            if intrinsic.type == IntrinsicType.MATH:
                headers.add("#include <metal_math>")
            elif intrinsic.type == IntrinsicType.ATOMIC:
                headers.add("#include <metal_atomic>")
            elif intrinsic.is_simd_function:
                headers.add("#include <metal_simdgroup>")
        return headers

    def get_vectorizable_intrinsics(self) -> Set[str]:
        """Get list of vectorizable intrinsic functions."""
        return {name for name, func in self.intrinsics.items() if func.vectorizable}

    def get_simd_functions(self) -> Set[str]:
        """Get list of SIMD-specific functions."""
        return {name for name, func in self.intrinsics.items() if func.is_simd_function}

    def validate_intrinsic_usage(self, node: dict) -> bool:
        """Validate intrinsic function usage."""
        func_name = node.get('function', {}).get('name')
        if not func_name or func_name not in self.intrinsics:
            return False

        intrinsic = self.intrinsics[func_name]
        return len(node.get('arguments', [])) == len(intrinsic.arg_types)

logger.info("IntrinsicFunctionMapper initialized with complete mappings")
