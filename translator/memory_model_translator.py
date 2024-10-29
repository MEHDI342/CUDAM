# memory_model_translator.py

from typing import Dict, List, Optional, Tuple, Union, Any
from enum import Enum
from dataclasses import dataclass
import logging

from ..utils.error_handler import CudaTranslationError
from ..utils.logger import get_logger

logger = get_logger(__name__)

class CudaMemorySpace(Enum):
    """CUDA memory spaces"""
    GLOBAL = "global"
    SHARED = "shared"
    CONSTANT = "constant"
    LOCAL = "local"
    TEXTURE = "texture"
    SURFACE = "surface"
    MANAGED = "managed"

class MetalMemorySpace(Enum):
    """Metal memory spaces"""
    DEVICE = "device"
    THREADGROUP = "threadgroup"
    CONSTANT = "constant"
    THREAD = "thread"
    TEXTURE = "texture"

@dataclass
class MemoryAccessPattern:
    """Represents memory access patterns for optimization"""
    is_coalesced: bool = False
    is_aligned: bool = False
    stride: Optional[int] = None
    vector_width: Optional[int] = None
    bank_conflicts: bool = False
    access_type: str = "random"
    thread_divergence: bool = False

class MemoryAlignment:
    """Memory alignment requirements"""
    METAL_MIN_ALIGNMENT = 16
    METAL_PREFERRED_ALIGNMENT = 256
    METAL_SIMD_WIDTH = 32
    METAL_WARP_SIZE = 32
    METAL_MAX_THREADGROUP_MEMORY = 32768  # 32KB

class MemoryModelTranslator:
    """
    Translates CUDA memory model concepts to Metal equivalents.
    Handles memory spaces, access patterns, and optimizations.
    """
    def __init__(self):
        self.memory_space_map = {
            CudaMemorySpace.GLOBAL: MetalMemorySpace.DEVICE,
            CudaMemorySpace.SHARED: MetalMemorySpace.THREADGROUP,
            CudaMemorySpace.CONSTANT: MetalMemorySpace.CONSTANT,
            CudaMemorySpace.LOCAL: MetalMemorySpace.THREAD,
            CudaMemorySpace.TEXTURE: MetalMemorySpace.TEXTURE
        }

        self.buffer_index_counter = 0
        self.threadgroup_memory_size = 0
        self.constant_memory_size = 0
        self.texture_index_counter = 0

    def translate_memory_space(self, cuda_space: CudaMemorySpace) -> MetalMemorySpace:
        """Translate CUDA memory space to Metal equivalent."""
        if cuda_space not in self.memory_space_map:
            raise CudaTranslationError(f"Unsupported CUDA memory space: {cuda_space}")
        return self.memory_space_map[cuda_space]

    def get_metal_declaration(self,
                              var_name: str,
                              data_type: str,
                              memory_space: MetalMemorySpace,
                              is_readonly: bool = False) -> str:
        """Generate Metal variable declaration with appropriate qualifiers."""
        qualifiers = []

        if memory_space == MetalMemorySpace.DEVICE:
            qualifier = "constant" if is_readonly else "device"
            buffer_index = self._get_next_buffer_index()
            qualifiers.append(f"{qualifier} {data_type}* {var_name} [[buffer({buffer_index})]]")
        elif memory_space == MetalMemorySpace.THREADGROUP:
            qualifiers.append(f"threadgroup {data_type} {var_name}")
        elif memory_space == MetalMemorySpace.CONSTANT:
            buffer_index = self._get_next_buffer_index()
            qualifiers.append(f"constant {data_type}& {var_name} [[buffer({buffer_index})]]")
        elif memory_space == MetalMemorySpace.THREAD:
            qualifiers.append(f"thread {data_type} {var_name}")
        elif memory_space == MetalMemorySpace.TEXTURE:
            texture_index = self._get_next_texture_index()
            access = "read" if is_readonly else "write"
            qualifiers.append(f"texture2d<float, access::{access}> {var_name} [[texture({texture_index})]]")

        return " ".join(qualifiers)

    def analyze_access_pattern(self,
                               indices: List[str],
                               thread_indices: List[str]) -> MemoryAccessPattern:
        """Analyze memory access pattern for optimization opportunities."""
        pattern = MemoryAccessPattern()

        # Check for coalesced access
        if any(idx in indices for idx in thread_indices):
            innermost_idx = indices[-1]
            if any(thread_idx in innermost_idx for thread_idx in thread_indices):
                pattern.is_coalesced = True

        # Calculate stride
        try:
            stride_expr = indices[-1]
            if stride_expr.isdigit():
                pattern.stride = int(stride_expr)

            # Check if power of 2 stride
            if pattern.stride and (pattern.stride & (pattern.stride - 1) == 0):
                pattern.is_aligned = True
        except (ValueError, AttributeError):
            pass

        # Check for vector access patterns
        if pattern.is_coalesced and pattern.is_aligned:
            if pattern.stride in (2, 4, 8, 16):
                pattern.vector_width = pattern.stride

        # Detect bank conflicts for threadgroup memory
        if len(indices) >= 2:
            potential_conflicts = self._analyze_bank_conflicts(indices)
            pattern.bank_conflicts = bool(potential_conflicts)

        # Determine access type
        if pattern.is_coalesced:
            pattern.access_type = "coalesced"
        elif pattern.stride == 1:
            pattern.access_type = "sequential"
        elif pattern.stride:
            pattern.access_type = "strided"

        return pattern

    def optimize_memory_layout(self,
                               size: int,
                               alignment: int,
                               access_pattern: MemoryAccessPattern) -> Tuple[int, int]:
        """Optimize memory layout based on access pattern."""
        # Ensure minimum alignment
        alignment = max(alignment, MemoryAlignment.METAL_MIN_ALIGNMENT)

        # Optimize for coalesced access
        if access_pattern.is_coalesced:
            alignment = max(alignment, MemoryAlignment.METAL_SIMD_WIDTH * 4)

        # Adjust for vector access
        if access_pattern.vector_width:
            alignment = max(alignment, access_pattern.vector_width * 4)

        # Round up size to alignment
        padded_size = (size + alignment - 1) & ~(alignment - 1)

        return padded_size, alignment

    def generate_buffer_bindings(self,
                                 variables: List[Tuple[str, str, CudaMemorySpace, bool]]) -> Dict[str, Any]:
        """Generate Metal buffer bindings for variables."""
        bindings = {}

        for var_name, data_type, memory_space, is_readonly in variables:
            metal_space = self.translate_memory_space(memory_space)

            if metal_space in (MetalMemorySpace.DEVICE, MetalMemorySpace.CONSTANT):
                buffer_index = self._get_next_buffer_index()
                bindings[var_name] = {
                    "index": buffer_index,
                    "type": data_type,
                    "space": metal_space.value,
                    "readonly": is_readonly
                }

        return bindings

    def allocate_threadgroup_memory(self, size: int) -> Optional[int]:
        """Allocate threadgroup memory and return offset."""
        if self.threadgroup_memory_size + size > MemoryAlignment.METAL_MAX_THREADGROUP_MEMORY:
            return None

        offset = self.threadgroup_memory_size
        self.threadgroup_memory_size += size
        return offset

    def generate_memory_barriers(self,
                                 scope: str,
                                 access_type: str = "all") -> str:
        """Generate appropriate Metal memory barriers."""
        if scope == "threadgroup":
            if access_type == "all":
                return "threadgroup_barrier(mem_flags::mem_threadgroup)"
            elif access_type == "read":
                return "threadgroup_barrier(mem_flags::mem_threadgroup_read)"
            elif access_type == "write":
                return "threadgroup_barrier(mem_flags::mem_threadgroup_write)"
        elif scope == "device":
            return "threadgroup_barrier(mem_flags::mem_device)"

        return ""

    def _get_next_buffer_index(self) -> int:
        """Get next available buffer index."""
        index = self.buffer_index_counter
        self.buffer_index_counter += 1
        return index

    def _get_next_texture_index(self) -> int:
        """Get next available texture index."""
        index = self.texture_index_counter
        self.texture_index_counter += 1
        return index

    def _analyze_bank_conflicts(self, indices: List[str]) -> List[Tuple[int, int]]:
        """Analyze potential bank conflicts in threadgroup memory access."""
        conflicts = []
        banks = {}

        for i, idx in enumerate(indices):
            try:
                bank = eval(idx) % 32  # Metal uses 32 banks
                if bank in banks:
                    conflicts.append((banks[bank], i))
                banks[bank] = i
            except:
                pass

        return conflicts

    def generate_texture_sampling(self,
                                  var_name: str,
                                  coords: List[str],
                                  sampler_state: Optional[Dict[str, Any]] = None) -> str:
        """Generate Metal texture sampling code."""
        if not sampler_state:
            sampler_state = {
                "filter": "linear",
                "address": "clamp_to_edge",
                "coord": "pixel"
            }

        sampler_desc = (f"sampler(filter::{sampler_state['filter']}, "
                        f"address::{sampler_state['address']}, "
                        f"coord::{sampler_state['coord']})")

        coord_expr = ", ".join(coords)
        return f"{var_name}.sample({sampler_desc}, float2({coord_expr}))"

    def get_optimal_threadgroup_size(self,
                                     work_items: int,
                                     memory_per_item: int) -> int:
        """Calculate optimal threadgroup size based on memory usage."""
        max_threads = min(
            MemoryAlignment.METAL_MAX_THREADGROUP_MEMORY // memory_per_item,
            1024  # Metal maximum threads per threadgroup
        )

        # Round down to multiple of SIMD width
        max_threads = (max_threads // MemoryAlignment.METAL_SIMD_WIDTH) * MemoryAlignment.METAL_SIMD_WIDTH

        # Find largest power of 2 <= max_threads that divides work_items evenly
        size = max_threads
        while size > 0:
            if work_items % size == 0:
                return size
            size -= MemoryAlignment.METAL_SIMD_WIDTH

        return MemoryAlignment.METAL_SIMD_WIDTH

logger.info("MemoryModelTranslator initialized for CUDA to Metal memory model translation.")