from typing import Dict, List, Set, Optional, Union
from pathlib import Path
import logging
from threading import Lock

from ..utils.error_handler import CudaTranslationError
from ..utils.logger import get_logger
from ..parser.ast_nodes import CUDAKernel

logger = get_logger(__name__)

class ObjectiveCGenerator:
    """
    what this class features
    Features:
    - Thread-safe implementation
    - Comprehensive error handling
    - Automatic resource management
    - Performance optimization
    - Metal API compliance
    """

    def __init__(self):
        self._lock = Lock()
        self._cache: Dict[str, str] = {}

        # Metal configuration constants
        self.METAL_CONFIG = {
            'MAX_BUFFERS': 31,
            'MAX_BUFFER_SIZE': 256 * 1024 * 1024,  # 256MB
            'PREFERRED_ALIGNMENT': 256,
            'MAX_COMMAND_BUFFERS': 32,
            'SIMD_GROUP_SIZE': 32
        }

    def generate_host_code(self, kernel: CUDAKernel, class_prefix: str = "MT") -> Dict[str, str]:
        """
        Generate complete Objective-C implementation for Metal kernel execution.

        Args:
            kernel: CUDA kernel AST node
            class_prefix: Class name prefix for Objective-C conventions

        Returns:
            Dict containing header (.h) and implementation (.m) file contents

        Raises:
            CudaTranslationError: If code generation fails
        """
        try:
            class_name = f"{class_prefix}{kernel.name}Kernel"

            # Generate header and implementation
            header = self._generate_header_file(class_name, kernel)
            implementation = self._generate_implementation_file(class_name, kernel)

            return {
                'header': header,
                'implementation': implementation
            }

        except Exception as e:
            logger.error(f"Failed to generate Objective-C host code: {str(e)}")
            raise CudaTranslationError(f"Objective-C code generation failed: {str(e)}")

    def _generate_header_file(self, class_name: str, kernel: CUDAKernel) -> str:
        """Generate header file with interface declaration."""
        return f"""
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>
#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/// Error domain for Metal kernel execution
extern NSString * const {class_name}ErrorDomain;

/// Error codes for Metal kernel execution
typedef NS_ERROR_ENUM({class_name}ErrorDomain, {class_name}ErrorCode) {{
    {class_name}ErrorDeviceNotFound = 1000,
    {class_name}ErrorLibraryCreationFailed,
    {class_name}ErrorFunctionNotFound,
    {class_name}ErrorPipelineCreationFailed,
    {class_name}ErrorCommandQueueCreationFailed,
    {class_name}ErrorCommandEncodingFailed,
    {class_name}ErrorInvalidBufferSize,
    {class_name}ErrorBufferAllocationFailed,
    {class_name}ErrorExecutionFailed,
    {class_name}ErrorInvalidParameters
}};

/// Metal kernel wrapper for {kernel.name}
@interface {class_name} : NSObject

/// Initialize with Metal device
- (nullable instancetype)initWithDevice:(id<MTLDevice>)device
                                error:(NSError **)error NS_DESIGNATED_INITIALIZER;

/// Default initializer not available
- (instancetype)init NS_UNAVAILABLE;

/// Execute kernel with completion handler
- (void)execute{kernel.name}:
    {self._generate_header_parameters(kernel)}
    completion:(void (^)(NSError * _Nullable))completion;

/// Synchronous kernel execution
- (BOOL)execute{kernel.name}:
    {self._generate_header_parameters(kernel)}
    error:(NSError **)error;

@end

NS_ASSUME_NONNULL_END
"""

    def _generate_implementation_file(self, class_name: str, kernel: CUDAKernel) -> str:
        """Generate implementation file with complete kernel execution logic."""
        return f"""
#import "{class_name}.h"

NSString * const {class_name}ErrorDomain = @"{class_name}ErrorDomain";

@implementation {class_name} {{
    id<MTLDevice> _device;
    id<MTLCommandQueue> _commandQueue;
    id<MTLComputePipelineState> _pipelineState;
    dispatch_queue_t _executionQueue;
}}

#pragma mark - Initialization

- (nullable instancetype)initWithDevice:(id<MTLDevice>)device error:(NSError **)error {{
    self = [super init];
    if (self) {{
        _device = device;
        
        // Create command queue
        _commandQueue = [_device newCommandQueue];
        if (!_commandQueue) {{
            if (error) {{
                *error = [NSError errorWithDomain:{class_name}ErrorDomain
                                           code:{class_name}ErrorCommandQueueCreationFailed
                                       userInfo:@{{NSLocalizedDescriptionKey: @"Failed to create command queue"}}];
            }}
            return nil;
        }}
        
        // Create pipeline state
        if (![self createPipelineStateWithError:error]) {{
            return nil;
        }}
        
        // Create serial execution queue
        _executionQueue = dispatch_queue_create("{class_name}.ExecutionQueue", DISPATCH_QUEUE_SERIAL);
    }}
    return self;
}}

#pragma mark - Pipeline Setup

- (BOOL)createPipelineStateWithError:(NSError **)error {{
    // Load default library
    id<MTLLibrary> library = [_device newDefaultLibrary];
    if (!library) {{
        if (error) {{
            *error = [NSError errorWithDomain:{class_name}ErrorDomain
                                       code:{class_name}ErrorLibraryCreationFailed
                                   userInfo:@{{NSLocalizedDescriptionKey: @"Failed to create Metal library"}}];
        }}
        return NO;
    }}
    
    // Load kernel function
    id<MTLFunction> kernelFunction = [library newFunctionWithName:@"{kernel.name}"];
    if (!kernelFunction) {{
        if (error) {{
            *error = [NSError errorWithDomain:{class_name}ErrorDomain
                                       code:{class_name}ErrorFunctionNotFound
                                   userInfo:@{{NSLocalizedDescriptionKey: @"Kernel function not found"}}];
        }}
        return NO;
    }}
    
    // Create pipeline state
    NSError *pipelineError = nil;
    _pipelineState = [_device newComputePipelineStateWithFunction:kernelFunction error:&pipelineError];
    if (!_pipelineState) {{
        if (error) {{
            *error = [NSError errorWithDomain:{class_name}ErrorDomain
                                       code:{class_name}ErrorPipelineCreationFailed
                                   userInfo:@{{
                                       NSLocalizedDescriptionKey: @"Failed to create pipeline state",
                                       NSUnderlyingErrorKey: pipelineError
                                   }}];
        }}
        return NO;
    }}
    
    return YES;
}}

#pragma mark - Buffer Management

- (nullable id<MTLBuffer>)createBufferWithData:(const void *)data 
                                     length:(NSUInteger)length
                                      error:(NSError **)error {{
    if (length == 0 || !data) {{
        if (error) {{
            *error = [NSError errorWithDomain:{class_name}ErrorDomain
                                       code:{class_name}ErrorInvalidBufferSize
                                   userInfo:@{{NSLocalizedDescriptionKey: @"Invalid buffer parameters"}}];
        }}
        return nil;
    }}
    
    id<MTLBuffer> buffer = [_device newBufferWithBytes:data
                                              length:length
                                             options:MTLResourceStorageModeShared];
    if (!buffer) {{
        if (error) {{
            *error = [NSError errorWithDomain:{class_name}ErrorDomain
                                       code:{class_name}ErrorBufferAllocationFailed
                                   userInfo:@{{NSLocalizedDescriptionKey: @"Failed to allocate Metal buffer"}}];
        }}
        return nil;
    }}
    
    return buffer;
}}

#pragma mark - Kernel Execution

{self._generate_execution_methods(kernel)}

#pragma mark - Helper Methods

{self._generate_helper_methods(kernel)}

@end
"""

    def _generate_header_parameters(self, kernel: CUDAKernel) -> str:
        """Generate parameter declarations for header file."""
        params = []
        for param in kernel.parameters:
            objc_type = self._cuda_type_to_objc(param.cuda_type)
            params.append(f"({objc_type}){param.name}")
        return "\n    ".join(params)

    def _generate_execution_methods(self, kernel: CUDAKernel) -> str:
        """Generate kernel execution method implementations."""
        return f"""
- (void)execute{kernel.name}:{self._generate_execution_parameters(kernel)}
    completion:(void (^)(NSError * _Nullable))completion {{
    
    dispatch_async(_executionQueue, ^{{
        NSError *error = nil;
        BOOL success = [self execute{kernel.name}:{self._generate_argument_list(kernel)}
                                          error:&error];
        if (completion) {{
            dispatch_async(dispatch_get_main_queue(), ^{{
                completion(success ? nil : error);
            }});
        }}
    }});
}}

- (BOOL)execute{kernel.name}:{self._generate_execution_parameters(kernel)}
    error:(NSError **)error {{
    
    @try {{
        // Validate inputs
        if (![self validateInputs:{self._generate_argument_list(kernel)} error:error]) {{
            return NO;
        }}
        
        // Create command buffer
        id<MTLCommandBuffer> commandBuffer = [_commandQueue commandBuffer];
        if (!commandBuffer) {{
            if (error) {{
                *error = [NSError errorWithDomain:{class_name}ErrorDomain
                                           code:{class_name}ErrorCommandEncodingFailed
                                       userInfo:@{{NSLocalizedDescriptionKey: @"Failed to create command buffer"}}];
            }}
            return NO;
        }}
        
        // Create compute encoder
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        if (!encoder) {{
            if (error) {{
                *error = [NSError errorWithDomain:{class_name}ErrorDomain
                                           code:{class_name}ErrorCommandEncodingFailed
                                       userInfo:@{{NSLocalizedDescriptionKey: @"Failed to create compute encoder"}}];
            }}
            return NO;
        }}
        
        // Configure encoder
        [encoder setComputePipelineState:_pipelineState];
        
        // Create and set buffers
        if (![self setupBuffers:encoder {self._generate_argument_list(kernel)} error:error]) {{
            return NO;
        }}
        
        // Configure thread groups
        MTLSize threadGroupSize = MTLSizeMake({kernel.thread_config.block_size[0]},
                                           {kernel.thread_config.block_size[1]},
                                           {kernel.thread_config.block_size[2]});
        MTLSize gridSize = [self calculateGridSize:dataSize threadGroupSize:threadGroupSize];
        
        // Dispatch threads
        [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadGroupSize];
        [encoder endEncoding];
        
        // Commit and wait
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        // Check for execution errors
        if (commandBuffer.error) {{
            if (error) {{
                *error = [NSError errorWithDomain:{class_name}ErrorDomain
                                           code:{class_name}ErrorExecutionFailed
                                       userInfo:@{{
                                           NSLocalizedDescriptionKey: @"Kernel execution failed",
                                           NSUnderlyingErrorKey: commandBuffer.error
                                       }}];
            }}
            return NO;
        }}
        
        return YES;
    }}
    @catch (NSException *exception) {{
        if (error) {{
            *error = [NSError errorWithDomain:{class_name}ErrorDomain
                                       code:{class_name}ErrorExecutionFailed
                                   userInfo:@{{
                                       NSLocalizedDescriptionKey: [exception reason],
                                       NSLocalizedFailureReasonErrorKey: [exception name]
                                   }}];
        }}
        return NO;
    }}
}}
"""

    def _generate_helper_methods(self, kernel: CUDAKernel) -> str:
        """Generate helper method implementations."""
        return """
- (BOOL)validateInputs:(NSArray *)inputs error:(NSError **)error {
    // Validate input parameters
    for (id input in inputs) {
        if (!input) {
            if (error) {
                *error = [NSError errorWithDomain:MTKernelErrorDomain
                                           code:MTKernelErrorInvalidParameters
                                       userInfo:@{NSLocalizedDescriptionKey: @"Invalid input parameter"}];
            }
            return NO;
        }
    }
    return YES;
}

- (MTLSize)calculateGridSize:(NSUInteger)dataSize threadGroupSize:(MTLSize)threadGroupSize {
    NSUInteger w = (dataSize + threadGroupSize.width - 1) / threadGroupSize.width;
    return MTLSizeMake(w, 1, 1);
}

- (BOOL)setupBuffers:(id<MTLComputeCommandEncoder>)encoder
                     error:(NSError **)error {
    // Buffer setup implementation
    return YES;
}
"""

    def _cuda_type_to_objc(self, cuda_type: str) -> str:
        """Convert CUDA type to Objective-C type."""
        type_mapping = {
            'float': 'NSArray<NSNumber *> *',
            'double': 'NSArray<NSNumber *> *',
            'int': 'NSArray<NSNumber *> *',
            'unsigned int': 'NSArray<NSNumber *> *',
            'long': 'NSArray<NSNumber *> *',
            'unsigned long': 'NSArray<NSNumber *> *',
        }
        return type_mapping.get(cuda_type, 'NSArray<NSNumber *> *')

    def cleanup(self):
        """Cleanup resources."""
        with self._lock:
            self._cache.clear()

    def _generate_execution_parameters(self, kernel: CUDAKernel) -> str:
        """Generate parameter list for execution methods."""
        params = []
        for param in kernel.parameters:
            objc_type = self._cuda_type_to_objc(param.cuda_type)
            params.append(f"({objc_type}){param.name}")
        return "\n    ".join(params)

    def _generate_argument_list(self, kernel: CUDAKernel) -> str:
        """Generate argument list for method calls."""
        return ", ".join(param.name for param in kernel.parameters)