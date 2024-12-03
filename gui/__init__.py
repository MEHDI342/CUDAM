# CUDAM/gui/__init__.py

"""
CUDAM GUI Module
Production-ready graphical interface for CUDA to Metal translation.
"""

import os
import sys
from pathlib import Path

# Add CUDAM root to Python path
CUDAM_ROOT = Path(__file__).parent.parent
if str(CUDAM_ROOT) not in sys.path:
    sys.path.append(str(CUDAM_ROOT))

# Core CUDAM components
from ..translator.host_adapter import HostAdapter
from ..translator.kernel_translator import KernelTranslator
from ..translator.cudnn_mapper import CudnnMapper
from ..translator.thread_hierarchy_mapper import ThreadHierarchyMapper

# Optimization components
from ..optimizer.unified_optimizer_metal import UnifiedMetalOptimizer
from ..optimization.memory_optimizer import MemoryOptimizer
from ..optimization.kernel_optimizer import KernelOptimizer
from ..optimization.barrier_optimizer import BarrierOptimizer

# Parser components
from ..parser.cuda_parser import CudaParser
from ..parser.cuda_syntax_validator import CudaSyntaxValidator
from ..parser.ast_nodes import (
    CUDANode, CUDAKernel, CUDAParameter, CUDAType, CUDAQualifier,
    CUDASharedMemory, CUDAThreadIdx, CUDABarrier
)

# Utility components
from ..utils.error_handler import (
    CudaError, CudaParseError, CudaTranslationError,
    CudaTypeError, CudaNotSupportedError
)
from ..utils.logger import get_logger
from ..utils.metal_equivalents import get_metal_equivalent
from ..utils.mapping_tables import MetalMappingRegistry

# Initialize logger
logger = get_logger(__name__)

class CudaTranslator:
    """
    Unified CUDA to Metal translation interface.
    Provides high-level access to translation functionality.
    """

    def __init__(self):
        self.parser = CudaParser()
        self.validator = CudaSyntaxValidator()
        self.kernel_translator = KernelTranslator()
        self.host_adapter = HostAdapter(self.kernel_translator, None)  # Initialize properly
        self.optimizer = UnifiedMetalOptimizer()
        self.mapping_registry = MetalMappingRegistry()

    def parse_cuda(self, cuda_code: str) -> CUDANode:
        """Parse CUDA code into AST."""
        try:
            return self.parser.parse_string(cuda_code)
        except CudaParseError as e:
            logger.error(f"Parsing error: {e}")
            raise

    def translate_to_metal(self, ast: CUDANode) -> str:
        """Translate CUDA AST to Metal code."""
        try:
            # Validate AST
            validation_result = self.validator.validate_ast(ast)
            if not validation_result[0]:
                raise CudaTranslationError(f"Validation failed: {validation_result[1]}")

            # Translate kernels
            metal_code = self.kernel_translator.translate_kernel(ast)

            # Adapt host code
            metal_code = self.host_adapter.translate_host_code(metal_code)

            return metal_code

        except Exception as e:
            logger.error(f"Translation error: {e}")
            raise CudaTranslationError(str(e))

    def optimize_metal_code(self, metal_code: str, optimization_level: int = 2) -> str:
        """Optimize generated Metal code."""
        try:
            return self.optimizer.optimize(metal_code, optimization_level)
        except Exception as e:
            logger.error(f"Optimization error: {e}")
            raise

# GUI-specific components
from .main import CUDAMMainWindow
from .widgets.editor import CodeEditor
from .widgets.project import ProjectExplorer
from .widgets.metrics import PerformanceMetrics

# Version information
__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Default configuration
DEFAULT_CONFIG = {
    'theme': 'light',
    'font_size': 12,
    'tab_size': 4,
    'auto_indent': True,
    'show_line_numbers': True,
    'highlight_current_line': True,
    'optimization_level': 2,
}

def initialize_gui():
    """Initialize GUI components and resources."""
    # Set up resource paths
    resource_path = Path(__file__).parent / 'resources'
    os.environ['CUDAM_RESOURCE_PATH'] = str(resource_path)

    # Initialize logging
    logger.info("Initializing CUDAM GUI")

    # Return initialized components
    return {
        'main_window': CUDAMMainWindow,
        'translator': CudaTranslator(),
        'config': DEFAULT_CONFIG
    }

# Expose public interface
__all__ = [
    'CUDAMMainWindow',
    'CudaTranslator',
    'initialize_gui',
    'CodeEditor',
    'ProjectExplorer',
    'PerformanceMetrics',
    '__version__',
]