from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging

from ..parser.ast_nodes import (
    CUDANode, CUDAType, CUDAKernel, CUDASharedMemory,
    CUDAThreadIdx, CUDABlockIdx
)

class MemoryAccessPattern(Enum):
    COALESCED = "coalesced"
    STRIDED = "strided"
    RANDOM = "random"
    BROADCAST = "broadcast"
    SEQUENTIAL = "sequential"

@dataclass
class MemoryAccess:
    """Information about a memory access"""
    node: CUDANode
    type: MemoryAccessPattern
    stride: Optional[int] = None
    scope: str = "global"
    is_read: bool = True
    is_atomic: bool = False
    alignment: int = 16
    vector_width: Optional[int] = None

class MemoryOptimizer:
    """
    Optimizes memory access patterns for Metal GPU following NVIDIA best practices
    """

    def __init__(self):
        self.simd_width = 32  # Metal SIMD width
        self.max_threads_per_group = 1024
        self.shared_memory_limit = 32768  # 32KB for Metal
        self.l1_cache_line_size = 128  # Metal cache line size
        self.vector_sizes = {2, 4, 8, 16}  # Supported vector widths
        self.memory_accesses: List[MemoryAccess] = []

    def optimize_kernel(self, kernel: CUDAKernel) -> CUDAKernel:
        """Apply memory optimizations to kernel"""
        # Analyze memory access patterns
        self._analyze_memory_accesses(kernel)

        # Apply optimizations
        kernel = self._optimize_global_memory(kernel)
        kernel = self._optimize_shared_memory(kernel)
        kernel = self._optimize_texture_memory(kernel)
        kernel = self._optimize_atomics(kernel)

        return kernel

    def _analyze_memory_accesses(self, kernel: CUDAKernel):
        """Analyze all memory accesses in kernel"""
        self.memory_accesses.clear()

        def visit_node(node: CUDANode):
            if access := self._detect_memory_access(node):
                self.memory_accesses.append(access)

        kernel.traverse(visit_node)

        # Group and analyze patterns
        self._analyze_access_patterns()

    def _detect_memory_access(self, node: CUDANode) -> Optional[MemoryAccess]:
        """Detect memory access type and pattern"""
        if not hasattr(node, 'cuda_type'):
            return None

        # Check for array access
        if self._is_array_access(node):
            pattern = self._determine_access_pattern(node)
            scope = self._determine_memory_scope(node)

            return MemoryAccess(
                node=node,
                type=pattern,
                scope=scope,
                stride=self._calculate_stride(node),
                vector_width=self._detect_vector_width(node),
                alignment=self._check_alignment(node)
            )

        return None

    def _is_array_access(self, node: CUDANode) -> bool:
        """Check if node represents array access"""
        return hasattr(node, 'is_pointer') and node.is_pointer

    def _determine_access_pattern(self, node: CUDANode) -> MemoryAccessPattern:
        """Determine memory access pattern"""
        thread_idx = self._find_thread_index(node)
        if not thread_idx:
            return MemoryAccessPattern.RANDOM

        # Check for coalesced access
        if self._is_coalesced_access(node, thread_idx):
            return MemoryAccessPattern.COALESCED

        # Check for strided access
        stride = self._calculate_stride(node)
        if stride:
            return MemoryAccessPattern.STRIDED

        # Check for broadcast
        if self._is_broadcast_access(node):
            return MemoryAccessPattern.BROADCAST

        return MemoryAccessPattern.RANDOM

    def _optimize_global_memory(self, kernel: CUDAKernel) -> CUDAKernel:
        """Optimize global memory access patterns"""
        coalescing_opportunities = [
            access for access in self.memory_accesses
            if access.scope == "global" and access.type != MemoryAccessPattern.COALESCED
        ]

        # Apply vectorization where possible
        for access in coalescing_opportunities:
            if self._can_vectorize(access):
                kernel = self._apply_vectorization(kernel, access)

        # Optimize array indexing
        kernel = self._optimize_array_indexing(kernel)

        # Add padding for alignment
        kernel = self._add_memory_padding(kernel)

        return kernel

    def _optimize_shared_memory(self, kernel: CUDAKernel) -> CUDAKernel:
        """Optimize shared memory usage"""
        shared_vars = [
            node for node in kernel.children
            if isinstance(node, CUDASharedMemory)
        ]

        total_size = 0
        for var in shared_vars:
            # Optimize bank conflicts
            var = self._resolve_bank_conflicts(var)

            # Track size
            size = self._calculate_shared_memory_size(var)
            total_size += size

            if total_size > self.shared_memory_limit:
                logging.warning(f"Shared memory usage {total_size} exceeds Metal limit {self.shared_memory_limit}")

        return kernel

    def _optimize_texture_memory(self, kernel: CUDAKernel) -> CUDAKernel:
        """Optimize texture memory usage"""
        # Find read-only array accesses that could use textures
        candidate_arrays = [
            access for access in self.memory_accesses
            if access.scope == "global" and access.is_read and not access.is_atomic
        ]

        for access in candidate_arrays:
            if self._should_use_texture(access):
                kernel = self._convert_to_texture(kernel, access)

        return kernel

    def _optimize_atomics(self, kernel: CUDAKernel) -> CUDAKernel:
        """Optimize atomic operations"""
        atomic_accesses = [
            access for access in self.memory_accesses
            if access.is_atomic
        ]

        for access in atomic_accesses:
            # Try to use simdgroup operations
            if self._can_use_simdgroup(access):
                kernel = self._convert_to_simdgroup(kernel, access)
            else:
                # Optimize atomic memory layout
                kernel = self._optimize_atomic_layout(kernel, access)

        return kernel

    def _resolve_bank_conflicts(self, shared_var: CUDASharedMemory) -> CUDASharedMemory:
        """Resolve shared memory bank conflicts"""
        if not self._has_bank_conflicts(shared_var):
            return shared_var

        # Add padding to avoid conflicts
        padding = self._calculate_padding(shared_var)
        shared_var.size += padding

        return shared_var

    def _calculate_padding(self, var: CUDASharedMemory) -> int:
        """Calculate padding to avoid bank conflicts"""
        type_size = self._get_type_size(var.cuda_type)
        banks = 32  # Metal uses 32 banks

        if var.size % banks == 0:
            return 0

        return banks - (var.size % banks)

    def _can_vectorize(self, access: MemoryAccess) -> bool:
        """Check if memory access can be vectorized"""
        if not access.stride:
            return False

        # Check if stride matches vector size
        return (
                access.stride in self.vector_sizes and
                access.alignment >= access.stride * 4 and  # 4 bytes per element
                not access.is_atomic
        )

    def _should_use_texture(self, access: MemoryAccess) -> bool:
        """Determine if array should use texture memory"""
        return (
                access.is_read and
                not access.is_atomic and
                access.type in {MemoryAccessPattern.RANDOM, MemoryAccessPattern.STRIDED} and
                self._get_type_size(access.node.cuda_type) <= 16  # Max texture element size
        )

    def _can_use_simdgroup(self, access: MemoryAccess) -> bool:
        """Check if atomic can use simdgroup operations"""
        return (
                access.is_atomic and
                access.type == MemoryAccessPattern.SEQUENTIAL and
                self._is_reduction_pattern(access)
        )

    def _get_type_size(self, cuda_type: CUDAType) -> int:
        """Get size of CUDA type in bytes"""
        size_map = {
            CUDAType.CHAR: 1,
            CUDAType.SHORT: 2,
            CUDAType.INT: 4,
            CUDAType.FLOAT: 4,
            CUDAType.DOUBLE: 8,
        }
        return size_map.get(cuda_type, 4)  # Default to 4 bytes

    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate memory optimization report"""
        return {
            "access_patterns": {
                pattern.value: len([a for a in self.memory_accesses if a.type == pattern])
                for pattern in MemoryAccessPattern
            },
            "vectorization_opportunities": len([
                a for a in self.memory_accesses if self._can_vectorize(a)
            ]),
            "texture_candidates": len([
                a for a in self.memory_accesses if self._should_use_texture(a)
            ]),
            "bank_conflicts": len([
                a for a in self.memory_accesses
                if a.scope == "shared" and self._has_bank_conflicts(a.node)
            ]),
            "simdgroup_opportunities": len([
                a for a in self.memory_accesses if self._can_use_simdgroup(a)
            ])
        }