
from typing import Dict, Any, Optional
import yaml
import json
import logging
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from ..utils.error_handler import CudaTranslationError
from ..utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class MetalConfig:
    """Metal-specific configuration settings."""
    max_threads_per_group: int = 1024
    max_total_threadgroup_memory: int = 32768  # 32KB
    simd_group_size: int = 32
    preferred_threadgroup_size: int = 256
    enable_fast_math: bool = True
    buffer_alignment: int = 256
    texture_alignment: int = 4096

@dataclass
class OptimizationConfig:
    """Optimization configuration settings."""
    level: int = 2
    enable_vectorization: bool = True
    enable_loop_unrolling: bool = True
    enable_memory_coalescing: bool = True
    enable_barrier_optimization: bool = True
    max_unroll_factor: int = 8
    cache_size: int = 32768
    thread_count: int = 4

@dataclass
class TranslationConfig:
    """Translation configuration settings."""
    target_language: str = "swift"
    generate_tests: bool = True
    preserve_comments: bool = True
    emit_debug_info: bool = True
    source_map: bool = True
    enable_profiling: bool = False
    inline_threshold: int = 100

class ConfigParser:
    """
    Advanced configuration parser with validation and optimization capabilities.
    Handles both YAML and JSON formats with extensive error checking.
    """

    def __init__(self):
        self.metal_config = MetalConfig()
        self.optimization_config = OptimizationConfig()
        self.translation_config = TranslationConfig()
        self.custom_mappings: Dict[str, Any] = {}
        self.validation_rules: Dict[str, Any] = {}

    def parse(self, config_path: str) -> Dict[str, Any]:
        """Parse and validate configuration file."""
        path = Path(config_path)

        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            config = self._load_config_file(path)
            self._validate_config(config)
            self._apply_config(config)
            return self._generate_final_config()
        except Exception as e:
            logger.error(f"Failed to parse configuration: {str(e)}")
            raise

    def _load_config_file(self, path: Path) -> Dict[str, Any]:
        """Load configuration from file with format detection."""
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

    def _validate_config(self, config: Dict[str, Any]):
        """Validate configuration with detailed error checking."""

        # Validate Metal configuration
        if 'metal' in config:
            self._validate_metal_config(config['metal'])

        # Validate optimization configuration
        if 'optimization' in config:
            self._validate_optimization_config(config['optimization'])

        # Validate translation configuration
        if 'translation' in config:
            self._validate_translation_config(config['translation'])

        # Validate custom mappings
        if 'mappings' in config:
            self._validate_custom_mappings(config['mappings'])

    def _validate_metal_config(self, config: Dict[str, Any]):
        """Validate Metal-specific configuration settings."""
        if 'max_threads_per_group' in config:
            value = config['max_threads_per_group']
            if not isinstance(value, int) or value <= 0 or value > 1024:
                raise ValueError("max_threads_per_group must be between 1 and 1024")

        if 'max_total_threadgroup_memory' in config:
            value = config['max_total_threadgroup_memory']
            if not isinstance(value, int) or value <= 0 or value > 32768:
                raise ValueError("max_total_threadgroup_memory must be between 1 and 32768")

    def _validate_optimization_config(self, config: Dict[str, Any]):
        """Validate optimization configuration settings."""
        if 'level' in config:
            level = config['level']
            if not isinstance(level, int) or level < 0 or level > 3:
                raise ValueError("Optimization level must be between 0 and 3")

        if 'thread_count' in config:
            count = config['thread_count']
            if not isinstance(count, int) or count < 1:
                raise ValueError("Thread count must be positive")

    def _validate_translation_config(self, config: Dict[str, Any]):
        """Validate translation configuration settings."""
        if 'target_language' in config:
            language = config['target_language'].lower()
            if language not in ['swift', 'objc']:
                raise ValueError("Target language must be 'swift' or 'objc'")

    def _validate_custom_mappings(self, mappings: Dict[str, Any]):
        """Validate custom type and function mappings."""
        if 'types' in mappings:
            self._validate_type_mappings(mappings['types'])
        if 'functions' in mappings:
            self._validate_function_mappings(mappings['functions'])

    def _apply_config(self, config: Dict[str, Any]):
        """Apply validated configuration to internal state."""
        with ThreadPoolExecutor() as executor:
            futures = []

            if 'metal' in config:
                futures.append(executor.submit(self._apply_metal_config, config['metal']))
            if 'optimization' in config:
                futures.append(executor.submit(self._apply_optimization_config, config['optimization']))
            if 'translation' in config:
                futures.append(executor.submit(self._apply_translation_config, config['translation']))
            if 'mappings' in config:
                futures.append(executor.submit(self._apply_custom_mappings, config['mappings']))

            # Wait for all configurations to be applied
            for future in futures:
                future.result()

    def _apply_metal_config(self, config: Dict[str, Any]):
        """Apply Metal configuration settings."""
        self.metal_config = MetalConfig(
            max_threads_per_group=config.get('max_threads_per_group', self.metal_config.max_threads_per_group),
            max_total_threadgroup_memory=config.get('max_total_threadgroup_memory', self.metal_config.max_total_threadgroup_memory),
            simd_group_size=config.get('simd_group_size', self.metal_config.simd_group_size),
            preferred_threadgroup_size=config.get('preferred_threadgroup_size', self.metal_config.preferred_threadgroup_size),
            enable_fast_math=config.get('enable_fast_math', self.metal_config.enable_fast_math),
            buffer_alignment=config.get('buffer_alignment', self.metal_config.buffer_alignment),
            texture_alignment=config.get('texture_alignment', self.metal_config.texture_alignment)
        )

    def _apply_optimization_config(self, config: Dict[str, Any]):
        """Apply optimization configuration settings."""
        self.optimization_config = OptimizationConfig(
            level=config.get('level', self.optimization_config.level),
            enable_vectorization=config.get('enable_vectorization', self.optimization_config.enable_vectorization),
            enable_loop_unrolling=config.get('enable_loop_unrolling', self.optimization_config.enable_loop_unrolling),
            enable_memory_coalescing=config.get('enable_memory_coalescing', self.optimization_config.enable_memory_coalescing),
            enable_barrier_optimization=config.get('enable_barrier_optimization', self.optimization_config.enable_barrier_optimization),
            max_unroll_factor=config.get('max_unroll_factor', self.optimization_config.max_unroll_factor),
            cache_size=config.get('cache_size', self.optimization_config.cache_size),
            thread_count=config.get('thread_count', self.optimization_config.thread_count)
        )

    def _apply_translation_config(self, config: Dict[str, Any]):
        """Apply translation configuration settings."""
        self.translation_config = TranslationConfig(
            target_language=config.get('target_language', self.translation_config.target_language),
            generate_tests=config.get('generate_tests', self.translation_config.generate_tests),
            preserve_comments=config.get('preserve_comments', self.translation_config.preserve_comments),
            emit_debug_info=config.get('emit_debug_info', self.translation_config.emit_debug_info),
            source_map=config.get('source_map', self.translation_config.source_map),
            enable_profiling=config.get('enable_profiling', self.translation_config.enable_profiling),
            inline_threshold=config.get('inline_threshold', self.translation_config.inline_threshold)
        )

    def _apply_custom_mappings(self, mappings: Dict[str, Any]):
        """Apply custom type and function mappings."""
        self.custom_mappings = mappings

    def _generate_final_config(self) -> Dict[str, Any]:
        """Generate final configuration dictionary."""
        return {
            'metal': {
                'max_threads_per_group': self.metal_config.max_threads_per_group,
                'max_total_threadgroup_memory': self.metal_config.max_total_threadgroup_memory,
                'simd_group_size': self.metal_config.simd_group_size,
                'preferred_threadgroup_size': self.metal_config.preferred_threadgroup_size,
                'enable_fast_math': self.metal_config.enable_fast_math,
                'buffer_alignment': self.metal_config.buffer_alignment,
                'texture_alignment': self.metal_config.texture_alignment
            },
            'optimization': {
                'level': self.optimization_config.level,
                'enable_vectorization': self.optimization_config.enable_vectorization,
                'enable_loop_unrolling': self.optimization_config.enable_loop_unrolling,
                'enable_memory_coalescing': self.optimization_config.enable_memory_coalescing,
                'enable_barrier_optimization': self.optimization_config.enable_barrier_optimization,
                'max_unroll_factor': self.optimization_config.max_unroll_factor,
                'cache_size': self.optimization_config.cache_size,
                'thread_count': self.optimization_config.thread_count
            },
            'translation': {
                'target_language': self.translation_config.target_language,
                'generate_tests': self.translation_config.generate_tests,
                'preserve_comments': self.translation_config.preserve_comments,
                'emit_debug_info': self.translation_config.emit_debug_info,
                'source_map': self.translation_config.source_map,
                'enable_profiling': self.translation_config.enable_profiling,
                'inline_threshold': self.translation_config.inline_threshold
            },
            'mappings': self.custom_mappings
        }

logger.info("ConfigParser initialized with Metal-specific optimizations.")