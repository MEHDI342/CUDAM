
from typing import Dict, Any, Optional, Tuple, List, Union, Callable
from enum import Enum
import math

class MetalMathFunction:
    """
    this cute class represent a what a metal math function should be with its properties and optimizations. kind of average but does the job still.
    """
    def __init__(self,
                 cuda_name: str,
                 metal_name: str,
                 return_type: str,
                 arg_types: List[str],
                 fast_variant: Optional[str] = None,
                 special_handling: Optional[Callable] = None,
                 accuracy_impact: str = "none",
                 performance_impact: str = "none"):
        self.cuda_name = cuda_name
        self.metal_name = metal_name
        self.return_type = return_type
        self.arg_types = arg_types
        self.fast_variant = fast_variant
        self.special_handling = special_handling
        self.accuracy_impact = accuracy_impact
        self.performance_impact = performance_impact

    def get_metal_declaration(self) -> str:
        """Generate Metal function declaration."""
        args = ', '.join(f"{t} x{i}" for i, t in enumerate(self.arg_types))
        return f"{self.return_type} {self.metal_name}({args})"

    def get_fast_variant_declaration(self) -> Optional[str]:
        """Generate fast variant declaration if available."""
        if not self.fast_variant:
            return None
        args = ', '.join(f"{t} x{i}" for i, t in enumerate(self.arg_types))
        return f"{self.return_type} {self.fast_variant}({args})"

class MetalMathAccuracy(Enum):
    """Defines accuracy levels for Metal math functions."""
    EXACT = "exact"
    APPROXIMATE = "approximate"
    REDUCED = "reduced"
    FAST = "fast"

class MetalMathOptimization:
    """Defines optimization strategies for math operations."""
    def __init__(self,
                 strategy: str,
                 conditions: List[Callable],
                 transformation: Callable,
                 performance_gain: float):
        self.strategy = strategy
        self.conditions = conditions
        self.transformation = transformation
        self.performance_gain = performance_gain

# Optimized special case handlers
def handle_pow_special_cases(base: Any, exponent: Any) -> Optional[str]:
    """Handle special cases for power function."""
    if isinstance(exponent, (int, float)):
        if exponent == 0.0:
            return "1.0"
        elif exponent == 1.0:
            return str(base)
        elif exponent == 2.0:
            return f"({base} * {base})"
        elif exponent == 0.5:
            return f"metal::sqrt({base})"
        elif exponent == -1.0:
            return f"1.0 / ({base})"
    return None

def handle_exp2_special_cases(x: Any) -> Optional[str]:
    """Handle special cases for exp2 function."""
    if isinstance(x, (int, float)):
        if x == 0.0:
            return "1.0"
        elif x == 1.0:
            return "2.0"
    return None

# Define Metal math function mappings
METAL_MATH_FUNCTIONS: Dict[str, MetalMathFunction] = {
    # Trigonometric Functions
    'sin': MetalMathFunction(
        cuda_name='sin',
        metal_name='metal::sin',
        return_type='float',
        arg_types=['float'],
        fast_variant='metal::fast::sin',
        accuracy_impact='reduced',
        performance_impact='improved'
    ),
    'cos': MetalMathFunction(
        cuda_name='cos',
        metal_name='metal::cos',
        return_type='float',
        arg_types=['float'],
        fast_variant='metal::fast::cos',
        accuracy_impact='reduced',
        performance_impact='improved'
    ),
    'tan': MetalMathFunction(
        cuda_name='tan',
        metal_name='metal::tan',
        return_type='float',
        arg_types=['float'],
        fast_variant='metal::fast::tan',
        accuracy_impact='reduced',
        performance_impact='improved'
    ),

    # Inverse Trigonometric Functions
    'asin': MetalMathFunction(
        cuda_name='asin',
        metal_name='metal::asin',
        return_type='float',
        arg_types=['float']
    ),
    'acos': MetalMathFunction(
        cuda_name='acos',
        metal_name='metal::acos',
        return_type='float',
        arg_types=['float']
    ),
    'atan': MetalMathFunction(
        cuda_name='atan',
        metal_name='metal::atan',
        return_type='float',
        arg_types=['float']
    ),
    'atan2': MetalMathFunction(
        cuda_name='atan2',
        metal_name='metal::atan2',
        return_type='float',
        arg_types=['float', 'float']
    ),

    # Hyperbolic Functions
    'sinh': MetalMathFunction(
        cuda_name='sinh',
        metal_name='metal::sinh',
        return_type='float',
        arg_types=['float']
    ),
    'cosh': MetalMathFunction(
        cuda_name='cosh',
        metal_name='metal::cosh',
        return_type='float',
        arg_types=['float']
    ),
    'tanh': MetalMathFunction(
        cuda_name='tanh',
        metal_name='metal::tanh',
        return_type='float',
        arg_types=['float']
    ),

    # Exponential and Logarithmic Functions
    'exp': MetalMathFunction(
        cuda_name='exp',
        metal_name='metal::exp',
        return_type='float',
        arg_types=['float'],
        fast_variant='metal::fast::exp',
        accuracy_impact='reduced',
        performance_impact='improved'
    ),
    'exp2': MetalMathFunction(
        cuda_name='exp2',
        metal_name='metal::exp2',
        return_type='float',
        arg_types=['float'],
        fast_variant='metal::fast::exp2',
        special_handling=handle_exp2_special_cases,
        accuracy_impact='reduced',
        performance_impact='improved'
    ),
    'log': MetalMathFunction(
        cuda_name='log',
        metal_name='metal::log',
        return_type='float',
        arg_types=['float'],
        fast_variant='metal::fast::log',
        accuracy_impact='reduced',
        performance_impact='improved'
    ),
    'log2': MetalMathFunction(
        cuda_name='log2',
        metal_name='metal::log2',
        return_type='float',
        arg_types=['float'],
        fast_variant='metal::fast::log2',
        accuracy_impact='reduced',
        performance_impact='improved'
    ),
    'log10': MetalMathFunction(
        cuda_name='log10',
        metal_name='metal::log10',
        return_type='float',
        arg_types=['float']
    ),

    # Power Functions
    'pow': MetalMathFunction(
        cuda_name='pow',
        metal_name='metal::pow',
        return_type='float',
        arg_types=['float', 'float'],
        fast_variant='metal::fast::pow',
        special_handling=handle_pow_special_cases,
        accuracy_impact='reduced',
        performance_impact='improved'
    ),
    'sqrt': MetalMathFunction(
        cuda_name='sqrt',
        metal_name='metal::sqrt',
        return_type='float',
        arg_types=['float'],
        fast_variant='metal::fast::sqrt',
        accuracy_impact='minimal',
        performance_impact='improved'
    ),
    'rsqrt': MetalMathFunction(
        cuda_name='rsqrt',
        metal_name='metal::rsqrt',
        return_type='float',
        arg_types=['float'],
        fast_variant='metal::fast::rsqrt',
        accuracy_impact='reduced',
        performance_impact='improved'
    ),

    # Rounding and Absolute Value Functions
    'ceil': MetalMathFunction(
        cuda_name='ceil',
        metal_name='metal::ceil',
        return_type='float',
        arg_types=['float']
    ),
    'floor': MetalMathFunction(
        cuda_name='floor',
        metal_name='metal::floor',
        return_type='float',
        arg_types=['float']
    ),
    'round': MetalMathFunction(
        cuda_name='round',
        metal_name='metal::round',
        return_type='float',
        arg_types=['float']
    ),
    'fabs': MetalMathFunction(
        cuda_name='fabs',
        metal_name='metal::abs',
        return_type='float',
        arg_types=['float']
    ),

    # Min/Max Functions
    'fmin': MetalMathFunction(
        cuda_name='fmin',
        metal_name='metal::min',
        return_type='float',
        arg_types=['float', 'float']
    ),
    'fmax': MetalMathFunction(
        cuda_name='fmax',
        metal_name='metal::max',
        return_type='float',
        arg_types=['float', 'float']
    ),

    # Misc Functions
    'fmod': MetalMathFunction(
        cuda_name='fmod',
        metal_name='metal::fmod',
        return_type='float',
        arg_types=['float', 'float']
    ),
    'remainder': MetalMathFunction(
        cuda_name='remainder',
        metal_name='metal::remainder',
        return_type='float',
        arg_types=['float', 'float']
    )
}

# Optimization patterns for common mathematical expressions
METAL_MATH_OPTIMIZATION_PATTERNS: Dict[str, MetalMathOptimization] = {
    'pow_to_mul': MetalMathOptimization(
        strategy="Convert pow to multiplication for integer exponents",
        conditions=[
            lambda node: node.function == 'pow',
            lambda node: isinstance(node.args[1], (int, float)),
            lambda node: node.args[1].is_integer() and 2 <= node.args[1] <= 4
        ],
        transformation=lambda base, exp: '*'.join([base] * int(exp)),
        performance_gain=2.0
    ),
    'exp2_to_shift': MetalMathOptimization(
        strategy="Convert exp2 to bit shift for integer inputs",
        conditions=[
            lambda node: node.function == 'exp2',
            lambda node: isinstance(node.args[0], (int, float)),
            lambda node: node.args[0].is_integer() and 0 <= node.args[0] <= 31
        ],
        transformation=lambda x: f"(1 << {int(x)})",
        performance_gain=3.0
    ),
    'sqrt_to_rsqrt': MetalMathOptimization(
        strategy="Convert sqrt to rsqrt when dividing by sqrt",
        conditions=[
            lambda node: node.parent.is_division(),
            lambda node: node.function == 'sqrt',
            lambda node: node.is_denominator()
        ],
        transformation=lambda x: f"metal::rsqrt({x})",
        performance_gain=1.5
    )
}

class MetalMathTranslator:
    """
    Translates CUDA math expressions to optimized Metal equivalents.
    """
    def __init__(self, optimization_level: int = 2):
        self.optimization_level = optimization_level
        self.functions = METAL_MATH_FUNCTIONS
        self.optimizations = METAL_MATH_OPTIMIZATION_PATTERNS

    def translate_function(self,
                           cuda_func: str,
                           args: List[Any],
                           fast_math: bool = False) -> str:
        """
        Translate a CUDA math function call to Metal.

        Args:
            cuda_func: Name of the CUDA function
            args: List of arguments
            fast_math: Whether to use fast math variants

        Returns:
            Metal function call as string
        """
        if cuda_func not in self.functions:
            raise ValueError(f"Unsupported CUDA math function: {cuda_func}")

        func = self.functions[cuda_func]

        # Check for special handling
        if func.special_handling:
            result = func.special_handling(*args)
            if result:
                return result

        # Apply optimizations if enabled
        if self.optimization_level > 0:
            for opt in self.optimizations.values():
                if all(condition(args) for condition in opt.conditions):
                    return opt.transformation(*args)

        # Use fast variant if enabled and available
        if fast_math and func.fast_variant:
            metal_func = func.fast_variant
        else:
            metal_func = func.metal_name

        # Generate function call
        arg_list = ', '.join(str(arg) for arg in args)
        return f"{metal_func}({arg_list})"

    def get_accuracy_impact(self, cuda_func: str, fast_math: bool = False) -> str:
        """Get accuracy impact of using this function."""
        if cuda_func not in self.functions:
            return "unknown"

        func = self.functions[cuda_func]
        if fast_math and func.fast_variant:
            return func.accuracy_impact
        return "none"

    def get_performance_impact(self, cuda_func: str, fast_math: bool = False) -> str:
        """Get performance impact of using this function."""
        if cuda_func not in self.functions:
            return "unknown"

        func = self.functions[cuda_func]
        if fast_math and func.fast_variant:
            return func.performance_impact
        return "none"

# Vector math function variants
METAL_VECTOR_MATH_FUNCTIONS = {
    'float2': {
        name: MetalMathFunction(
            cuda_name=f'{name}f2',
            metal_name=f'metal::{name}',
            return_type='float2',
            arg_types=['float2'],
            fast_variant=f'metal::fast::{name}' if func.fast_variant else None,
            accuracy_impact=func.accuracy_impact,
            performance_impact=func.performance_impact
        )
        for name, func in METAL_MATH_FUNCTIONS.items()
        if len(func.arg_types) == 1
    },
    'float3': {
        name: MetalMathFunction(
            cuda_name=f'{name}f3',
            metal_name=f'metal::{name}',
            return_type='float3',
            arg_types=['float3'],
            fast_variant=f'metal::fast::{name}' if func.fast_variant else None,
            accuracy_impact=func.accuracy_impact,
            performance_impact=func.performance_impact
        )
        for name, func in METAL_MATH_FUNCTIONS.items()
        if len(func.arg_types) == 1
    },
    'float4': {
        name: MetalMathFunction(
            cuda_name=f'{name}f4',
            metal_name=f'metal::{name}',
            return_type='float4',
            arg_types=['float4'],
            fast_variant=f'metal::fast::{name}' if func.fast_variant else None,
            accuracy_impact=func.accuracy_impact,
            performance_impact=func.performance_impact
        )
        for name, func in METAL_MATH_FUNCTIONS.items()
        if len(func.arg_types) == 1
    }
}

# SIMD optimized variants for Metal
METAL_SIMD_MATH_FUNCTIONS = {
    name: MetalMathFunction(
        cuda_name=f'simd_{name}',
        metal_name=f'metal::simd::{name}',
        return_type='simd_float4',
        arg_types=['simd_float4'],
        fast_variant=f'metal::simd::fast::{name}' if func.fast_variant else None,
        accuracy_impact=func.accuracy_impact,
        performance_impact='highly_improved'
    )
    for name, func in METAL_MATH_FUNCTIONS.items()
    if len(func.arg_types) == 1
}

class MetalMathRegistry:
    """
    Registry for Metal math functions with optimization capabilities.
    """
    def __init__(self):
        self.scalar_functions = METAL_MATH_FUNCTIONS
        self.vector_functions = METAL_VECTOR_MATH_FUNCTIONS
        self.simd_functions = METAL_SIMD_MATH_FUNCTIONS
        self.optimization_patterns = METAL_MATH_OPTIMIZATION_PATTERNS

    def get_function(self,
                     name: str,
                     vector_size: Optional[int] = None,
                     use_simd: bool = False) -> Optional[MetalMathFunction]:
        """Get the appropriate Metal math function."""
        if use_simd and name in self.simd_functions:
            return self.simd_functions[name]

        if vector_size:
            vector_type = f'float{vector_size}'
            if vector_type in self.vector_functions and name in self.vector_functions[vector_type]:
                return self.vector_functions[vector_type][name]

        return self.scalar_functions.get(name)

    def get_optimization_pattern(self, name: str) -> Optional[MetalMathOptimization]:
        """Get an optimization pattern by name."""
        return self.optimization_patterns.get(name)

    def register_custom_function(self,
                                 function: MetalMathFunction,
                                 vector_sizes: Optional[List[int]] = None):
        """Register a custom math function."""
        self.scalar_functions[function.cuda_name] = function

        if vector_sizes:
            for size in vector_sizes:
                vector_type = f'float{size}'
                if vector_type not in self.vector_functions:
                    self.vector_functions[vector_type] = {}
                self.vector_functions[vector_type][function.cuda_name] = MetalMathFunction(
                    cuda_name=f'{function.cuda_name}f{size}',
                    metal_name=function.metal_name,
                    return_type=f'float{size}',
                    arg_types=[f'float{size}' for _ in function.arg_types],
                    fast_variant=function.fast_variant,
                    accuracy_impact=function.accuracy_impact,
                    performance_impact=function.performance_impact
                )

    def register_optimization_pattern(self,
                                      name: str,
                                      pattern: MetalMathOptimization):
        """Register a custom optimization pattern."""
        self.optimization_patterns[name] = pattern

# Initialize global registry
metal_math_registry = MetalMathRegistry()

def get_metal_math_registry() -> MetalMathRegistry:
    """Get the global Metal math registry instance."""
    return metal_math_registry