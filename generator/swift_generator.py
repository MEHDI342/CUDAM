from typing import Dict, List, Set, Optional, Union
from pathlib import Path
import logging
from threading import Lock

from ..utils.error_handler import CudaTranslationError
from ..utils.logger import get_logger
from ..parser.ast_nodes import CUDAKernel

logger = get_logger(__name__)

class SwiftGenerator:
    """
    Production-grade Swift code generator for Metal kernel integration.
    Handles host-side code generation with proper memory management and error handling.
    """

    def __init__(self):
        self._lock = Lock()
        self._cache: Dict[str, str] = {}

        # Metal-specific settings
        self.metal_settings = {
            'max_buffers': 31,
            'max_buffer_size': 256 * 1024 * 1024,  # 256MB
            'preferred_alignment': 256,
            'max_command_buffers': 32
        }

    def generate_host_code(self, kernel: CUDAKernel, class_name: Optional[str] = None) -> str:
        """Generate Swift host code for Metal kernel execution."""
        try:
            # Generate core components
            class_name = class_name or f"{kernel.name}Kernel"
            imports = self._generate_imports()
            class_def = self._generate_class_definition(class_name, kernel)
            buffer_management = self._generate_buffer_management(kernel)
            kernel_execution = self._generate_kernel_execution(kernel)
            error_handling = self._generate_error_handling()

            # Combine all components
            swift_code = f"""
{imports}

// MARK: - Metal Kernel Implementation
{class_def}

    // MARK: - Properties
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let pipelineState: MTLComputePipelineState
    private var buffers: [String: MTLBuffer] = [:]

    // MARK: - Initialization
    init() throws {{
        guard let device = MTLCreateSystemDefaultDevice() else {{
            throw MetalError.deviceNotFound
        }}
        self.device = device

        guard let commandQueue = device.makeCommandQueue() else {{
            throw MetalError.commandQueueCreationFailed
        }}
        self.commandQueue = commandQueue

        self.pipelineState = try Self.createPipelineState(device: device)
    }}

    // MARK: - Pipeline Setup
    private static func createPipelineState(device: MTLDevice) throws -> MTLComputePipelineState {{
        guard let library = device.makeDefaultLibrary() else {{
            throw MetalError.libraryCreationFailed
        }}

        guard let kernelFunction = library.makeFunction(name: "{kernel.name}") else {{
            throw MetalError.functionNotFound
        }}

        do {{
            return try device.makeComputePipelineState(function: kernelFunction)
        }} catch {{
            throw MetalError.pipelineCreationFailed
        }}
    }}

{buffer_management}

    // MARK: - Kernel Execution
{kernel_execution}

{error_handling}
}}

// MARK: - Extension for Async/Await Support
extension {class_name} {{
    /// Execute kernel with async/await support
    func executeAsync(
        {self._generate_parameter_list(kernel)}
    ) async throws {{
        try await withCheckedThrowingContinuation {{ continuation in
            execute(
                {self._generate_argument_list(kernel)},
                completion: {{ result in
                    switch result {{
                    case .success:
                        continuation.resume()
                    case .failure(let error):
                        continuation.resume(throwing: error)
                    }}
                }}
            )
        }}
    }}

    /// Execute kernel with completion handler
    func execute(
        {self._generate_parameter_list(kernel)},
        completion: @escaping (Result<Void, Error>) -> Void
    ) {{
        do {{
            // Validate input parameters
            try validateInputs({self._generate_validation_list(kernel)})

            // Create command buffer and encoder
            guard let commandBuffer = commandQueue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else {{
                throw MetalError.commandEncodingFailed
            }}

            // Configure encoder
            encoder.setComputePipelineState(pipelineState)
            
            // Set buffers
            try setBuffers(encoder: encoder, {self._generate_buffer_list(kernel)})

            // Calculate optimal thread configuration
            let threadGroupSize = MTLSize(width: {kernel.thread_config.block_size[0]},
                                        height: {kernel.thread_config.block_size[1]},
                                        depth: {kernel.thread_config.block_size[2]})
            let gridSize = calculateGridSize(dataSize: dataSize, threadGroupSize: threadGroupSize)

            // Dispatch threads
            encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadGroupSize)
            encoder.endEncoding()

            // Add completion handler
            commandBuffer.addCompletedHandler {{ buffer in
                if let error = buffer.error {{
                    completion(.failure(MetalError.executionFailed(error)))
                }} else {{
                    completion(.success(()))
                }}
            }}

            // Commit command buffer
            commandBuffer.commit()

        }} catch {{
            completion(.failure(error))
        }}
    }}

    // MARK: - Private Helper Methods
    private func validateInputs({self._generate_parameter_list(kernel)}) throws {{
        // Implement input validation logic based on kernel requirements
        {self._generate_validation_code(kernel)}
    }}

    private func setBuffers(
        encoder: MTLComputeCommandEncoder,
        {self._generate_parameter_list(kernel)}
    ) throws {{
        // Set buffers with proper error handling
        {self._generate_buffer_setup_code(kernel)}
    }}

    private func calculateGridSize(dataSize: Int, threadGroupSize: MTLSize) -> MTLSize {{
        let w = (dataSize + threadGroupSize.width - 1) / threadGroupSize.width
        return MTLSizeMake(w, 1, 1)
    }}
}}

// MARK: - Error Types
enum MetalError: LocalizedError {{
    case deviceNotFound
    case libraryCreationFailed
    case functionNotFound
    case pipelineCreationFailed
    case commandQueueCreationFailed
    case commandEncodingFailed
    case invalidBufferSize
    case bufferAllocationFailed
    case executionFailed(Error)
    case invalidInputParameters(String)

    var errorDescription: String? {{
        switch self {{
        case .deviceNotFound:
            return "Metal device not found"
        case .libraryCreationFailed:
            return "Failed to create Metal library"
        case .functionNotFound:
            return "Metal kernel function not found"
        case .pipelineCreationFailed:
            return "Failed to create compute pipeline state"
        case .commandQueueCreationFailed:
            return "Failed to create command queue"
        case .commandEncodingFailed:
            return "Failed to create command encoder"
        case .invalidBufferSize:
            return "Invalid buffer size specified"
        case .bufferAllocationFailed:
            return "Failed to allocate Metal buffer"
        case .executionFailed(let error):
            return "Kernel execution failed: \\(error.localizedDescription)"
        case .invalidInputParameters(let message):
            return "Invalid input parameters: \\(message)"
        }}
    }}
}}

// MARK: - Buffer Management Extension
private extension {class_name} {{
    func createBuffer<T>(from data: [T], options: MTLResourceOptions = .storageModeShared) throws -> MTLBuffer {{
        let size = MemoryLayout<T>.stride * data.count
        guard size > 0 else {{
            throw MetalError.invalidBufferSize
        }}

        guard let buffer = device.makeBuffer(bytes: data,
                                           length: size,
                                           options: options) else {{
            throw MetalError.bufferAllocationFailed
        }}

        return buffer
    }}

    func createBuffer<T>(size: Int, options: MTLResourceOptions = .storageModeShared) throws -> MTLBuffer {{
        guard size > 0 else {{
            throw MetalError.invalidBufferSize
        }}

        guard let buffer = device.makeBuffer(length: size,
                                           options: options) else {{
            throw MetalError.bufferAllocationFailed
        }}

        return buffer
    }}
}}
"""

            return swift_code

        except Exception as e:
            logger.error(f"Failed to generate Swift host code: {str(e)}")
            raise CudaTranslationError(f"Swift code generation failed: {str(e)}")

    def _generate_imports(self) -> str:
        """Generate required import statements."""
        return """
import Metal
import MetalKit
import Foundation
"""

    def _generate_class_definition(self, class_name: str, kernel: CUDAKernel) -> str:
        """Generate class definition with documentation."""
        return f"""
/// Metal kernel wrapper for {kernel.name}
/// Provides type-safe interface for kernel execution with proper error handling
final class {class_name} {{"""

    def _generate_parameter_list(self, kernel: CUDAKernel) -> str:
        """Generate parameter list for function signatures."""
        params = []
        for param in kernel.parameters:
            swift_type = self._cuda_type_to_swift(param.cuda_type)
            params.append(f"{param.name}: {swift_type}")
        return ", ".join(params)

    def _generate_validation_code(self, kernel: CUDAKernel) -> str:
        """Generate input validation code."""
        validations = []
        for param in kernel.parameters:
            if param.is_buffer:
                validations.append(f"""
        if {param.name}.count == 0 {{
            throw MetalError.invalidInputParameters("Empty buffer for {param.name}")
        }}""")
        return "\n".join(validations)

    def _generate_buffer_setup_code(self, kernel: CUDAKernel) -> str:
        """Generate buffer setup code."""
        setups = []
        for idx, param in enumerate(kernel.parameters):
            if param.is_buffer:
                setups.append(f"""
        let {param.name}Buffer = try createBuffer(from: {param.name})
        encoder.setBuffer({param.name}Buffer, offset: 0, index: {idx})""")
        return "\n".join(setups)

    def _cuda_type_to_swift(self, cuda_type: str) -> str:
        """Convert CUDA type to Swift type."""
        type_mapping = {
            'float': '[Float]',
            'double': '[Double]',
            'int': '[Int32]',
            'unsigned int': '[UInt32]',
            'long': '[Int64]',
            'unsigned long': '[UInt64]',
        }
        return type_mapping.get(cuda_type, '[Float]')  # Default to [Float] if type not found

    def cleanup(self):
        """Cleanup any resources."""
        with self._lock:
            self._cache.clear()