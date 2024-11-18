from typing import Dict, List, Optional, Set
from dataclasses import dataclass
import re

from ..parser.ast_nodes import (
    CUDANode, CUDAKernel, CUDAParameter, CUDAType, CUDAQualifier,
    CUDASharedMemory, CUDAThreadIdx, CUDABlockIdx, CUDAGridDim,
    CUDAAtomicOperation, CUDASync
)

@dataclass
class MetalKernelInfo:
    """Metadata for Metal kernel generation"""
    name: str
    buffer_bindings: Dict[str, int]
    threadgroup_memory_size: int
    simd_group_size: int = 32
    max_threads_per_group: int = 1024

class MetalGenerator:
    """Generates Metal Shading Language code from CUDA kernels"""

    def __init__(self):
        self.kernel_info: Dict[str, MetalKernelInfo] = {}
        self.next_buffer_index: int = 0
        self.required_metal_features: Set[str] = set()

    def generate_metal_code(self, kernel: CUDAKernel) -> str:
        """Generate Metal kernel code from CUDA kernel"""
        self.kernel_info[kernel.name] = self._create_kernel_info(kernel)

        # Generate kernel components
        signature = self._generate_kernel_signature(kernel)
        body = self._generate_kernel_body(kernel)

        # Combine components
        metal_code = []
        metal_code.extend(self._generate_includes())
        metal_code.append(signature)
        metal_code.append("{")
        metal_code.extend(body)
        metal_code.append("}")

        return "\n".join(metal_code)

    def _create_kernel_info(self, kernel: CUDAKernel) -> MetalKernelInfo:
        """Create Metal kernel information"""
        buffer_bindings = {}

        # Assign buffer indices to parameters
        for param in kernel.parameters:
            buffer_bindings[param.name] = self.next_buffer_index
            self.next_buffer_index += 1

        return MetalKernelInfo(
            name=kernel.name,
            buffer_bindings=buffer_bindings,
            threadgroup_memory_size=self._calculate_shared_memory(kernel)
        )

    def _calculate_shared_memory(self, kernel: CUDAKernel) -> int:
        """Calculate total shared memory usage"""
        total_size = 0
        for node in kernel.children:
            if isinstance(node, CUDASharedMemory) and node.size:
                type_size = self._get_type_size(node.cuda_type)
                total_size += type_size * node.size
        return total_size

    def _generate_includes(self) -> List[str]:
        """Generate required Metal includes"""
        includes = [
            "#include <metal_stdlib>",
            "#include <metal_atomic>",
            "#include <metal_math>",
            "#include <metal_simdgroup>",
            "",
            "using namespace metal;"
        ]
        return includes

    def _generate_kernel_signature(self, kernel: CUDAKernel) -> str:
        """Generate Metal kernel signature"""
        params = []
        kernel_info = self.kernel_info[kernel.name]

        # Convert parameters to Metal
        for param in kernel.parameters:
            metal_type = self._cuda_to_metal_type(param.cuda_type)
            if param.is_pointer:
                qualifier = "device" if not param.is_readonly else "const device"
                buffer_idx = kernel_info.buffer_bindings[param.name]
                params.append(
                    f"{qualifier} {metal_type}* {param.name} [[buffer({buffer_idx})]]"
                )
            else:
                buffer_idx = kernel_info.buffer_bindings[param.name]
                params.append(
                    f"constant {metal_type}& {param.name} [[buffer({buffer_idx})]]"
                )

        # Add thread position parameters
        params.extend([
            "uint3 threadIdx [[thread_position_in_threadgroup]]",
            "uint3 blockIdx [[threadgroup_position_in_grid]]",
            "uint3 blockDim [[threads_per_threadgroup]]",
            "uint3 gridDim [[threadgroups_per_grid]]"
        ])

        return f"kernel void {kernel.name}({', '.join(params)})"

    def _generate_kernel_body(self, kernel: CUDAKernel) -> List[str]:
        """Generate Metal kernel body"""
        body = []

        # Add shared memory declarations
        shared_mem = self._generate_shared_memory_declarations(kernel)
        if shared_mem:
            body.extend(shared_mem)
            body.append("")

        # Process kernel contents
        for node in kernel.children:
            if code := self._generate_node_code(node):
                body.extend(code)

        return ["    " + line for line in body if line]

    def _generate_shared_memory_declarations(self, kernel: CUDAKernel) -> List[str]:
        """Generate Metal threadgroup memory declarations"""
        declarations = []

        for node in kernel.children:
            if isinstance(node, CUDASharedMemory):
                metal_type = self._cuda_to_metal_type(node.cuda_type)
                if node.size:
                    declarations.append(
                        f"threadgroup {metal_type} {node.name}[{node.size}];"
                    )
                else:
                    declarations.append(
                        f"threadgroup {metal_type} {node.name};"
                    )

        return declarations

    def _generate_node_code(self, node: CUDANode) -> List[str]:
        """Generate Metal code for a specific node type"""
        if isinstance(node, CUDAThreadIdx):
            return [f"threadIdx.{node.dimension}"]

        elif isinstance(node, CUDABlockIdx):
            return [f"blockIdx.{node.dimension}"]

        elif isinstance(node, CUDAGridDim):
            return [f"gridDim.{node.dimension}"]

        elif isinstance(node, CUDAAtomicOperation):
            return self._generate_atomic_operation(node)

        elif isinstance(node, CUDASync):
            return self._generate_sync(node)

        return []

    def _generate_atomic_operation(self, node: CUDAAtomicOperation) -> List[str]:
        """Generate Metal atomic operation"""
        atomic_map = {
            'Add': 'atomic_fetch_add_explicit',
            'Sub': 'atomic_fetch_sub_explicit',
            'Exch': 'atomic_exchange_explicit',
            'Min': 'atomic_fetch_min_explicit',
            'Max': 'atomic_fetch_max_explicit',
            'And': 'atomic_fetch_and_explicit',
            'Or': 'atomic_fetch_or_explicit',
            'Xor': 'atomic_fetch_xor_explicit',
            'CAS': 'atomic_compare_exchange_weak_explicit'
        }

        metal_func = atomic_map.get(node.operation)
        if metal_func:
            args = [arg.get_metal_translation() for arg in node.children]
            args.append("memory_order_relaxed")
            return [f"{metal_func}({', '.join(args)});"]

        return []

    def _generate_sync(self, node: CUDASync) -> List[str]:
        """Generate Metal synchronization"""
        sync_map = {
            'syncthreads': 'threadgroup_barrier(mem_flags::mem_threadgroup)',
            'threadfence': 'threadgroup_barrier(mem_flags::mem_device)',
            'threadfence_block': 'threadgroup_barrier(mem_flags::mem_threadgroup)',
            'threadfence_system': 'threadgroup_barrier(mem_flags::mem_device)',
        }

        return [f"{sync_map.get(node.sync_type)};"]

    def _cuda_to_metal_type(self, cuda_type: CUDAType) -> str:
        """Map CUDA type to Metal type"""
        type_map = {
            CUDAType.VOID: 'void',
            CUDAType.CHAR: 'char',
            CUDAType.UCHAR: 'uchar',
            CUDAType.SHORT: 'short',
            CUDAType.USHORT: 'ushort',
            CUDAType.INT: 'int',
            CUDAType.UINT: 'uint',
            CUDAType.LONG: 'int64_t',
            CUDAType.ULONG: 'uint64_t',
            CUDAType.FLOAT: 'float',
            CUDAType.DOUBLE: 'float',  # Metal doesn't support double

            # Vector types
            CUDAType.INT2: 'int2',
            CUDAType.INT3: 'int3',
            CUDAType.INT4: 'int4',
            CUDAType.UINT2: 'uint2',
            CUDAType.UINT3: 'uint3',
            CUDAType.UINT4: 'uint4',
            CUDAType.FLOAT2: 'float2',
            CUDAType.FLOAT3: 'float3',
            CUDAType.FLOAT4: 'float4',
        }

        return type_map.get(cuda_type, 'void')

    def _get_type_size(self, cuda_type: CUDAType) -> int:
        """Get size in bytes of CUDA type"""
        size_map = {
            CUDAType.CHAR: 1,
            CUDAType.UCHAR: 1,
            CUDAType.SHORT: 2,
            CUDAType.USHORT: 2,
            CUDAType.INT: 4,
            CUDAType.UINT: 4,
            CUDAType.LONG: 8,
            CUDAType.ULONG: 8,
            CUDAType.FLOAT: 4,
            CUDAType.DOUBLE: 8,

            # Vector types
            CUDAType.INT2: 8,
            CUDAType.INT3: 12,
            CUDAType.INT4: 16,
            CUDAType.UINT2: 8,
            CUDAType.UINT3: 12,
            CUDAType.UINT4: 16,
            CUDAType.FLOAT2: 8,
            CUDAType.FLOAT3: 12,
            CUDAType.FLOAT4: 16,
        }

        return size_map.get(cuda_type, 0)