from typing import Dict, Any
import re
from pathlib import Path

from ..utils.error_handler import CudaTranslationError
from ..utils.logger import get_logger
from ..core.parser.ast_nodes import (
    CUDANode, CUDAKernel, CUDAParameter, CUDAType,
    CUDAQualifier, CUDASharedMemory, CUDAThreadIdx
)
from ..generator.msl_generator import MetalShaderGenerator


class CUDAHostTranslator:
    """
    Translates CUDA host code to Metal host code following NVIDIA's host API patterns
    """

    def __init__(self):
        self.metal_buffer_index = 0
        self.kernel_map: Dict[str, CUDAKernel] = {}

    def translate_host_code(self, cuda_code: str, target_lang: str = 'swift') -> str:
        """Translate CUDA host code to Metal"""
        if target_lang not in {'swift', 'objc'}:
            raise ValueError("Target language must be 'swift' or 'objc'")

        # Process CUDA API calls
        processed_code = self._translate_device_management(cuda_code)
        processed_code = self._translate_memory_management(processed_code)
        processed_code = self._translate_kernel_launch(processed_code)
        processed_code = self._translate_synchronization(processed_code)

        # Generate appropriate host code
        if target_lang == 'swift':
            return self._generate_swift_code(processed_code)
        else:
            return self._generate_objc_code(processed_code)

    def _translate_device_management(self, code: str) -> str:
        """Translate CUDA device management calls"""
        replacements = {
            r'cudaSetDevice\((\d+)\)': r'// Metal automatically manages devices',
            r'cudaGetDevice\(&dev\)': r'// Metal automatically manages devices',
            r'cudaGetDeviceCount\(&count\)': r'let count = MTLCopyAllDevices().count',
            r'cudaDeviceSynchronize\(\)': r'commandBuffer.waitUntilCompleted()'
        }

        result = code
        for cuda_pattern, metal_code in replacements.items():
            result = re.sub(cuda_pattern, metal_code, result)

        return result

    def _translate_memory_management(self, code: str) -> str:
        """Translate CUDA memory management calls"""
        # Handle cudaMalloc
        code = re.sub(
            r'cudaMalloc\(\(void\*\*\)&(\w+),\s*(.+?)\)',
            lambda m: f'{m.group(1)} = device.makeBuffer(length: {m.group(2)}, '
                      f'options: .storageModeShared)',
            code
        )

        # Handle cudaMemcpy
        code = re.sub(
            r'cudaMemcpy\((.+?),\s*(.+?),\s*(.+?),\s*cudaMemcpy(.+?)\)',
            self._translate_memcpy,
            code
        )

        # Handle cudaFree
        code = re.sub(
            r'cudaFree\((\w+)\)',
            r'// Metal automatically manages memory',
            code
        )

        return code

    def _translate_memcpy(self, match) -> str:
        """Translate cudaMemcpy calls"""
        dst, src, size, kind = match.groups()

        if kind == 'HostToDevice':
            return f'memcpy({dst}.contents, {src}, {size})'
        elif kind == 'DeviceToHost':
            return f'memcpy({dst}, {src}.contents, {size})'
        elif kind == 'DeviceToDevice':
            return (f'let blitEncoder = commandBuffer.makeBlitCommandEncoder()\n'
                    f'blitEncoder.copy(from: {src}, to: {dst}, size: {size})\n'
                    f'blitEncoder.endEncoding()')

        return match.group(0)

    def _translate_kernel_launch(self, code: str) -> str:
        """Translate CUDA kernel launches"""
        # Match kernel launch syntax
        pattern = r'(\w+)<<<(.+?)>>>(.+?);'

        return re.sub(pattern, self._translate_launch_config, code)

    def _translate_launch_config(self, match) -> str:
        """Translate kernel launch configuration"""
        kernel_name, config, args = match.groups()

        # Parse grid and block dimensions
        grid_dim, block_dim = config.split(',', 1)

        return (
            f'let commandEncoder = commandBuffer.makeComputeCommandEncoder()\n'
            f'commandEncoder.setComputePipelineState({kernel_name}PipelineState)\n'
            f'let gridSize = MTLSize(width: {grid_dim}, height: 1, depth: 1)\n'
            f'let blockSize = MTLSize(width: {block_dim}, height: 1, depth: 1)\n'
            f'commandEncoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: blockSize)\n'
            f'commandEncoder.endEncoding()'
        )

    def _translate_synchronization(self, code: str) -> str:
        """Translate CUDA synchronization calls"""
        replacements = {
            r'cudaDeviceSynchronize\(\)': 'commandBuffer.waitUntilCompleted()',
            r'cudaStreamSynchronize\((\w+)\)': r'\1.waitUntilCompleted()',
            r'cudaEventSynchronize\((\w+)\)': r'\1.waitUntilCompleted()',
        }

        result = code
        for cuda_pattern, metal_code in replacements.items():
            result = re.sub(cuda_pattern, metal_code, result)

        return result

    def _generate_swift_code(self, processed_code: str) -> str:
        """Generate Swift host code"""
        setup_code = """
            import Metal
            import MetalKit
            
            guard let device = MTLCreateSystemDefaultDevice() else {
                fatalError("GPU not available")
            }
            
            let commandQueue = device.makeCommandQueue()!
            let commandBuffer = commandQueue.makeCommandBuffer()!
        """

        return f"{setup_code}\n{processed_code}"

    def _generate_objc_code(self, processed_code: str) -> str:
        """Generate Objective-C host code"""
        setup_code = """
            #import <Metal/Metal.h>
            #import <MetalKit/MetalKit.h>
            
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (!device) {
                NSLog(@"GPU not available");
                return;
            }
            
            id<MTLCommandQueue> commandQueue = [device newCommandQueue];
            id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        """

        return f"{setup_code}\n{processed_code}"