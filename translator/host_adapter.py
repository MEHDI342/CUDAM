import re
from typing import Dict, Any
from ..utils.error_handler import CudaTranslationError
from ..utils.logger import get_logger
from ..translator.kernel_translator import KernelTranslator
from ..translator.memory_model_translator import MemoryModelTranslator

logger = get_logger(__name__)

class HostAdapter:
    def __init__(self, kernel_translator: KernelTranslator, memory_translator: MemoryModelTranslator):
        self.kernel_translator = kernel_translator
        self.memory_translator = memory_translator
        self.cuda_to_metal_api = {
            'cudaMalloc': 'newBufferWithLength',
            'cudaFree': None,
            'cudaMemcpy': 'contents',
            'cudaStreamCreate': 'newCommandQueue',
            'cudaStreamDestroy': None,
            'cudaEventCreate': 'newEvent',
            'cudaEventRecord': 'enqueue',
            'cudaEventSynchronize': 'waitUntilCompleted',
            'cudaDeviceSynchronize': 'commit'
        }

    def translate_host_code(self, cuda_code: str) -> str:
        metal_code = cuda_code

        for cuda_api, metal_api in self.cuda_to_metal_api.items():
            if metal_api:
                metal_code = metal_code.replace(cuda_api, metal_api)
            else:
                metal_code = self.remove_unsupported_call(metal_code, cuda_api)

        metal_code = self.adapt_kernel_launches(metal_code)
        metal_code = self.translate_memory_management(metal_code)
        return metal_code

    def remove_unsupported_call(self, code: str, api_call: str) -> str:
        pattern = rf'{api_call}\s*\([^)]*\);'
        return re.sub(pattern, f'// Removed unsupported CUDA call: {api_call}', code)

    def adapt_kernel_launches(self, code: str) -> str:
        kernel_launch_pattern = r'(\w+)<<<(.+?)>>>(.+?);'

        def replace_kernel_launch(match):
            kernel_name = match.group(1)
            launch_params = match.group(2).split(',')
            kernel_args = match.group(3)

            grid_dim = launch_params[0].strip()
            block_dim = launch_params[1].strip()

            return f"""
            MTLSize gridSize = MTLSizeMake({grid_dim}, 1, 1);
            MTLSize threadGroupSize = MTLSizeMake({block_dim}, 1, 1);
            [commandEncoder setComputePipelineState:{kernel_name}PipelineState];
            [commandEncoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadGroupSize];
            {self.kernel_translator.translate_kernel(kernel_name)}{kernel_args};
            """

        return re.sub(kernel_launch_pattern, replace_kernel_launch, code)

    def translate_memory_management(self, code: str) -> str:
        malloc_pattern = r'cudaMalloc\(\(void\*\*\)&(\w+),\s*(.+?)\);'
        code = re.sub(malloc_pattern, lambda m: f"{m.group(1)} = [device newBufferWithLength:{m.group(2)} options:MTLResourceStorageModeShared];", code)

        memcpy_pattern = r'cudaMemcpy\((.+?),\s*(.+?),\s*(.+?),\s*cudaMemcpy(.+?)\);'
        code = re.sub(memcpy_pattern, lambda m: f"memcpy({m.group(1)}.contents, {m.group(2)}, {m.group(3)});", code)

        return code

    def generate_metal_setup(self) -> str:
        return """
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> commandQueue = [device newCommandQueue];
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];
        """

    def generate_metal_cleanup(self) -> str:
        return """
        [commandEncoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        """

logger.info("HostAdapter initialized for CUDA to Metal host code translation.")