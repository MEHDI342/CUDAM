"""
Configuration Parser - Complete Production Implementation
Handles parsing and validation of Metal configuration settings
"""

from dataclasses import dataclass
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from ..utils.error_handler import CudaTranslationError
from ..utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class MetalConfig:
    """Metal-specific configuration settings"""
    max_threads_per_group: int = 1024
    max_total_threadgroup_memory: int = 32768  # 32KB
    simd_group_size: int = 32
    preferred_threadgroup_size: int = 256
    enable_fast_math: bool = True
    buffer_alignment: int = 256
    texture_alignment: int = 4096

@dataclass
class OptimizationConfig:
    """Optimization configuration settings"""
    level: int = 2
    enable_vectorization: bool = True
    enable_loop_unrolling: bool = True
    enable_memory_coalescing: bool = True
    enable_barrier_optimization: bool = True
    max_unroll_factor: int = 8
    cache_size: int = 32768
    thread_count: int = 4

class ConfigParser:
    """
    Thread-safe configuration parser with validation and optimization support.
    Handles both YAML and JSON formats with extensive error checking.
    """

    def __init__(self):
        """Initialize parser with default configurations"""
        self.metal_config = MetalConfig()
        self.optimization_config = OptimizationConfig()
        self._lock = Lock()
        self._cache: Dict[str, Any] = {}

        # Thread pool for parallel validation
        self._executor = ThreadPoolExecutor(max_workers=4)

    def parse(self, config_path: str) -> Dict[str, Any]:
        """
        Parse and validate configuration file.

        Args:
            config_path: Path to configuration file

        Returns:
            Dict containing validated configuration

        Raises:
            CudaTranslationError: If configuration is invalid
        """
        try:
            path = Path(config_path)
            if not path.exists():
                raise CudaTranslationError(f"Configuration file not found: {config_path}")

            # Load and parse configuration
            config = self._load_config_file(path)

            # Validate configuration
            self._validate_configuration(config)

            # Apply configuration
            with self._lock:
                self._apply_configuration(config)

            return self._generate_final_config()

        except Exception as e:
            logger.error(f"Failed to parse configuration: {e}")
            raise CudaTranslationError(f"Configuration parsing failed: {str(e)}")

    def _load_config_file(self, path: Path) -> Dict[str, Any]:
        """Load configuration from file with format detection"""
        content = path.read_text()

        if path.suffix in ['.yaml', '.yml']:
            try:
                return yaml.safe_load(content)
            except yaml.YAMLError as e:
                raise CudaTranslationError(f"Invalid YAML configuration: {str(e)}")
        elif path.suffix == '.json':
            try:
                return json.loads(content)
            except json.JSONDecodeError as e:
                raise CudaTranslationError(f"Invalid JSON configuration: {str(e)}")
        else:
            raise CudaTranslationError(f"Unsupported configuration format: {path.suffix}")

    def _validate_configuration(self, config: Dict[str, Any]):
        """Validate all configuration sections"""
        futures = []

        with self._executor as executor:
            if 'metal' in config:
                futures.append(
                    executor.submit(self._validate_metal_config, config['metal'])
                )
            if 'optimization' in config:
                futures.append(
                    executor.submit(self._validate_optimization_config, config['optimization'])
                )
            if 'translation' in config:
                futures.append(
                    executor.submit(self._validate_translation_config, config['translation'])
                )

        # Check validation results
        for future in futures:
            future.result()  # This will raise any validation errors

    def _validate_metal_config(self, config: Dict[str, Any]):
        """Validate Metal configuration parameters with hardware constraints"""
    if 'max_threads_per_group' in config:
        value = config['max_threads_per_group']
        if not isinstance(value, int) or value <= 0 or value > 1024:
            raise CudaTranslationError(
                f"max_threads_per_group must be between 1 and 1024, got {value}"
            )

    if 'max_total_threadgroup_memory' in config:
        value = config['max_total_threadgroup_memory']
        if not isinstance(value, int) or value <= 0 or value > 32768:
            raise CudaTranslationError(
                f"max_total_threadgroup_memory must be between 1 and 32768, got {value}"
            )

    if 'simd_group_size' in config:
        value = config['simd_group_size']
        if value != 32:  # Metal requires SIMD group size of 32
            raise CudaTranslationError("simd_group_size must be 32 for Metal")

    self._validate_memory_alignment(config)
    self._validate_thread_dimensions(config)

def _validate_memory_alignment(self, config: Dict[str, Any]):
    """Validate memory alignment requirements"""
    for param in ['buffer_alignment', 'texture_alignment']:
        if param in config:
            value = config[param]
            if not isinstance(value, int) or value <= 0 or (value & (value - 1)) != 0:
                raise CudaTranslationError(
                    f"{param} must be a positive power of 2, got {value}"
                )

def _validate_thread_dimensions(self, config: Dict[str, Any]):
    """Validate thread dimension constraints"""
    if 'preferred_threadgroup_size' in config:
        size = config['preferred_threadgroup_size']
        if not isinstance(size, int) or size <= 0 or size > 1024:
            raise CudaTranslationError(
                f"preferred_threadgroup_size must be between 1 and 1024, got {size}"
            )
        if size % 32 != 0:  # Must be multiple of SIMD width
            raise CudaTranslationError(
                f"preferred_threadgroup_size must be multiple of 32, got {size}"
            )

def _validate_optimization_config(self, config: Dict[str, Any]):
    """Validate optimization settings with performance implications"""
    if 'level' in config:
        level = config['level']
        if not isinstance(level, int) or level < 0 or level > 3:
            raise CudaTranslationError(
                f"Optimization level must be between 0 and 3, got {level}"
            )

    for bool_param in [
        'enable_vectorization',
        'enable_loop_unrolling',
        'enable_memory_coalescing',
        'enable_barrier_optimization'
    ]:
        if bool_param in config and not isinstance(config[bool_param], bool):
            raise CudaTranslationError(
                f"{bool_param} must be a boolean value"
            )

    self._validate_optimization_factors(config)
    self._validate_resource_limits(config)

def _validate_optimization_factors(self, config: Dict[str, Any]):
    """Validate optimization factor constraints"""
    if 'max_unroll_factor' in config:
        factor = config['max_unroll_factor']
        if not isinstance(factor, int) or factor <= 0 or factor > 32:
            raise CudaTranslationError(
                f"max_unroll_factor must be between 1 and 32, got {factor}"
            )

    if 'cache_size' in config:
        size = config['cache_size']
        if not isinstance(size, int) or size <= 0:
            raise CudaTranslationError(
                f"cache_size must be positive, got {size}"
            )

def _validate_resource_limits(self, config: Dict[str, Any]):
    """Validate hardware resource limitations"""
    if 'thread_count' in config:
        count = config['thread_count']
        if not isinstance(count, int) or count <= 0:
            raise CudaTranslationError(
                f"thread_count must be positive, got {count}"
            )

        # Check system CPU count for reasonable limits
        import os
        cpu_count = os.cpu_count() or 1
        if count > cpu_count * 4:
            logger.warning(
                f"thread_count {count} exceeds recommended maximum of {cpu_count * 4}"
            )

def _apply_configuration(self, config: Dict[str, Any]):
    """Apply validated configuration settings"""
    with self._lock:
        if 'metal' in config:
            self._apply_metal_config(config['metal'])
        if 'optimization' in config:
            self._apply_optimization_config(config['optimization'])
        if 'translation' in config:
            self._apply_translation_config(config['translation'])

def _apply_metal_config(self, config: Dict[str, Any]):
    """Apply Metal-specific configuration"""
    self.metal_config = MetalConfig(
        max_threads_per_group=config.get(
            'max_threads_per_group',
            self.metal_config.max_threads_per_group
        ),
        max_total_threadgroup_memory=config.get(
            'max_total_threadgroup_memory',
            self.metal_config.max_total_threadgroup_memory
        ),
        simd_group_size=config.get(
            'simd_group_size',
            self.metal_config.simd_group_size
        ),
        preferred_threadgroup_size=config.get(
            'preferred_threadgroup_size',
            self.metal_config.preferred_threadgroup_size
        ),
        enable_fast_math=config.get(
            'enable_fast_math',
            self.metal_config.enable_fast_math
        ),
        buffer_alignment=config.get(
            'buffer_alignment',
            self.metal_config.buffer_alignment
        ),
        texture_alignment=config.get(
            'texture_alignment',
            self.metal_config.texture_alignment
        )
    )

def _apply_optimization_config(self, config: Dict[str, Any]):
    """Apply optimization configuration"""
    self.optimization_config = OptimizationConfig(
        level=config.get('level', self.optimization_config.level),
        enable_vectorization=config.get(
            'enable_vectorization',
            self.optimization_config.enable_vectorization
        ),
        enable_loop_unrolling=config.get(
            'enable_loop_unrolling',
            self.optimization_config.enable_loop_unrolling
        ),
        enable_memory_coalescing=config.get(
            'enable_memory_coalescing',
            self.optimization_config.enable_memory_coalescing
        ),
        enable_barrier_optimization=config.get(
            'enable_barrier_optimization',
            self.optimization_config.enable_barrier_optimization
        ),
        max_unroll_factor=config.get(
            'max_unroll_factor',
            self.optimization_config.max_unroll_factor
        ),
        cache_size=config.get(
            'cache_size',
            self.optimization_config.cache_size
        ),
        thread_count=config.get(
            'thread_count',
            self.optimization_config.thread_count
        )
    )

def _generate_final_config(self) -> Dict[str, Any]:
    """Generate final configuration dictionary"""
    return {
        'metal': {
            'max_threads_per_group': self.metal_config.max_threads_per_group,
            'max_total_threadgroup_memory':
                self.metal_config.max_total_threadgroup_memory,
            'simd_group_size': self.metal_config.simd_group_size,
            'preferred_threadgroup_size':
                self.metal_config.preferred_threadgroup_size,
            'enable_fast_math': self.metal_config.enable_fast_math,
            'buffer_alignment': self.metal_config.buffer_alignment,
            'texture_alignment': self.metal_config.texture_alignment
        },
        'optimization': {
            'level': self.optimization_config.level,
            'enable_vectorization': self.optimization_config.enable_vectorization,
            'enable_loop_unrolling': self.optimization_config.enable_loop_unrolling,
            'enable_memory_coalescing':
                self.optimization_config.enable_memory_coalescing,
            'enable_barrier_optimization':
                self.optimization_config.enable_barrier_optimization,
            'max_unroll_factor': self.optimization_config.max_unroll_factor,
            'cache_size': self.optimization_config.cache_size,
            'thread_count': self.optimization_config.thread_count
        }
    }

def cleanup(self):
    """Clean up resources"""
    try:
        self._executor.shutdown(wait=True)
        with self._lock:
            self._cache.clear()
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")