from typing import Dict, Tuple, Any
from ..utils.error_handler import CudaTranslationError
from ..utils.logger import get_logger

logger = get_logger(__name__)

class ThreadHierarchyMapper:
    def __init__(self):
        self.cuda_to_metal_map = {
            'threadIdx': 'thread_position_in_threadgroup',
            'blockIdx': 'threadgroup_position_in_grid',
            'blockDim': 'threadgroup_size',
            'gridDim': 'grid_size'
        }
        self.max_threads_per_threadgroup = 1024  # This may vary depending on the Metal device

    def map_thread_id(self, cuda_expr: str) -> str:
        for cuda_var, metal_var in self.cuda_to_metal_map.items():
            if cuda_var in cuda_expr:
                return cuda_expr.replace(cuda_var, metal_var)
        raise CudaTranslationError(f"Unsupported CUDA thread hierarchy expression: {cuda_expr}")

    def calculate_global_id(self, dim: str) -> str:
        return f"(thread_position_in_threadgroup.{dim} + (threadgroup_position_in_grid.{dim} * threadgroup_size.{dim}))"

    def translate_launch_parameters(self, grid_dim: Tuple[int, int, int], block_dim: Tuple[int, int, int]) -> Dict[str, Any]:
        optimized_grid_dim, optimized_block_dim = self.optimize_thread_hierarchy(grid_dim, block_dim)
        return {
            'threads_per_threadgroup': self._create_metal_size(optimized_block_dim),
            'threadgroups_per_grid': self._create_metal_size(optimized_grid_dim)
        }

    def _create_metal_size(self, dim: Tuple[int, int, int]) -> str:
        return f"MTLSizeMake({dim[0]}, {dim[1]}, {dim[2]})"

    def generate_metal_dispatch(self, kernel_name: str, grid_dim: Tuple[int, int, int], block_dim: Tuple[int, int, int]) -> str:
        launch_params = self.translate_launch_parameters(grid_dim, block_dim)
        return f"""
        [commandEncoder setComputePipelineState:{kernel_name}PipelineState];
        [commandEncoder dispatchThreadgroups:{launch_params['threadgroups_per_grid']}
                        threadsPerThreadgroup:{launch_params['threads_per_threadgroup']}];
        """

    def translate_shared_memory(self, cuda_shared_mem: str) -> str:
        return cuda_shared_mem.replace("__shared__", "threadgroup")

    def translate_syncthreads(self) -> str:
        return "threadgroup_barrier(metal::mem_flags::mem_threadgroup);"

    def translate_block_sync(self) -> str:
        return "threadgroup_barrier(metal::mem_flags::mem_device);"

    def translate_grid_sync(self) -> str:
        logger.warning("Grid-wide synchronization is not directly supported in Metal. Using device memory barrier.")
        return "threadgroup_barrier(metal::mem_flags::mem_device);"

    def optimize_thread_hierarchy(self, grid_dim: Tuple[int, int, int], block_dim: Tuple[int, int, int]) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        total_threads = block_dim[0] * block_dim[1] * block_dim[2]
        if total_threads > self.max_threads_per_threadgroup:
            scale_factor = (self.max_threads_per_threadgroup / total_threads) ** (1/3)
            new_block_dim = tuple(int(dim * scale_factor) for dim in block_dim)
            new_grid_dim = tuple(int(grid_dim[i] * (block_dim[i] / new_block_dim[i])) for i in range(3))
            return new_grid_dim, new_block_dim

        # Ensure block dimensions are multiples of the SIMD width (usually 32 for Metal GPUs)
        simd_width = 32
        optimized_block_dim = tuple(((dim + simd_width - 1) // simd_width) * simd_width for dim in block_dim)

        # Adjust grid dimensions to account for changes in block dimensions
        optimized_grid_dim = tuple((grid_dim[i] * block_dim[i] + optimized_block_dim[i] - 1) // optimized_block_dim[i] for i in range(3))

        return optimized_grid_dim, optimized_block_dim

    def translate_warp_level_operations(self, cuda_expr: str) -> str:
        warp_ops = {
            '__shfl': 'simd_shuffle',
            '__shfl_up': 'simd_shuffle_up',
            '__shfl_down': 'simd_shuffle_down',
            '__shfl_xor': 'simd_shuffle_xor',
            '__all': 'simd_all',
            '__any': 'simd_any',
            '__ballot': 'simd_ballot'
        }
        for cuda_op, metal_op in warp_ops.items():
            if cuda_op in cuda_expr:
                return cuda_expr.replace(cuda_op, metal_op)
        return cuda_expr

    def adjust_kernel_launch(self, kernel_name: str, grid_dim: Tuple[int, int, int], block_dim: Tuple[int, int, int]) -> str:
        optimized_grid_dim, optimized_block_dim = self.optimize_thread_hierarchy(grid_dim, block_dim)
        return self.generate_metal_dispatch(kernel_name, optimized_grid_dim, optimized_block_dim)

logger.info("ThreadHierarchyMapper initialized for CUDA to Metal thread hierarchy translation.")