#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>
#import "kernel_wrapper.h"

// CUDA-style error codes
typedef NS_ENUM(NSInteger, CUDAError) {
    cudaSuccess = 0,
    cudaErrorDeviceNotFound = 1,
    cudaErrorMemoryAllocation = 2,
    cudaErrorInvalidValue = 3,
    cudaErrorLaunchFailure = 4
};

@implementation CUDAMetalDevice {
    id<MTLDevice> _device;
    id<MTLCommandQueue> _commandQueue;
    NSMutableDictionary<NSString*, id<MTLComputePipelineState>>* _kernelPipelineStates;
    NSMutableDictionary<NSString*, id<MTLFunction>>* _kernelFunctions;
    NSMutableDictionary* _allocatedBuffers;
}

- (instancetype)init {
    self = [super init];
    if (self) {
        _device = MTLCreateSystemDefaultDevice();
        if (!_device) {
            return nil;
        }

        _commandQueue = [_device newCommandQueue];
        if (!_commandQueue) {
            return nil;
        }

        _kernelPipelineStates = [NSMutableDictionary new];
        _kernelFunctions = [NSMutableDictionary new];
        _allocatedBuffers = [NSMutableDictionary new];
    }
    return self;
}

// CUDA Memory Management
- (CUDAError)cudaMalloc:(void**)ptr size:(size_t)size {
    id<MTLBuffer> buffer = [_device newBufferWithLength:size
                                              options:MTLResourceStorageModeShared];
    if (!buffer) {
        return cudaErrorMemoryAllocation;
    }

    *ptr = buffer.contents;
    [_allocatedBuffers setObject:buffer forKey:[NSValue valueWithPointer:*ptr]];

    return cudaSuccess;
}

- (CUDAError)cudaFree:(void*)ptr {
    [_allocatedBuffers removeObjectForKey:[NSValue valueWithPointer:ptr]];
    return cudaSuccess;
}

- (CUDAError)cudaMemcpy:(void*)dst
                   src:(const void*)src
                  size:(size_t)size
                  kind:(CUDAMemcpyKind)kind {
    switch (kind) {
        case cudaMemcpyHostToDevice: {
            id<MTLBuffer> buffer = [_allocatedBuffers objectForKey:[NSValue valueWithPointer:dst]];
            if (!buffer) return cudaErrorInvalidValue;
            memcpy(buffer.contents, src, size);
            break;
        }

        case cudaMemcpyDeviceToHost: {
            id<MTLBuffer> buffer = [_allocatedBuffers objectForKey:[NSValue valueWithPointer:src]];
            if (!buffer) return cudaErrorInvalidValue;
            memcpy(dst, buffer.contents, size);
            break;
        }

        case cudaMemcpyDeviceToDevice: {
            id<MTLBuffer> srcBuffer = [_allocatedBuffers objectForKey:[NSValue valueWithPointer:src]];
            id<MTLBuffer> dstBuffer = [_allocatedBuffers objectForKey:[NSValue valueWithPointer:dst]];
            if (!srcBuffer || !dstBuffer) return cudaErrorInvalidValue;

            id<MTLCommandBuffer> commandBuffer = [_commandQueue commandBuffer];
            id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];

            [blitEncoder copyFromBuffer:srcBuffer
                         sourceOffset:0
                             toBuffer:dstBuffer
                    destinationOffset:0
                                size:size];

            [blitEncoder endEncoding];
                        [commandBuffer commit];
                        [commandBuffer waitUntilCompleted];
                        break;
                    }
                }
                return cudaSuccess;
            }

            // Kernel Management
            - (CUDAError)loadMetalLibraryWithURL:(NSURL*)url error:(NSError**)error {
                id<MTLLibrary> library = [_device newLibraryWithURL:url error:error];
                if (!library) {
                    return cudaErrorLaunchFailure;
                }

                // Load all kernel functions
                for (NSString* functionName in library.functionNames) {
                    id<MTLFunction> function = [library newFunctionWithName:functionName];
                    if (!function) continue;

                    _kernelFunctions[functionName] = function;

                    // Create pipeline state
                    id<MTLComputePipelineState> pipelineState =
                        [_device newComputePipelineStateWithFunction:function error:error];
                    if (pipelineState) {
                        _kernelPipelineStates[functionName] = pipelineState;
                    }
                }

                return cudaSuccess;
            }

            // CUDA Kernel Launch
            - (CUDAError)launchKernel:(NSString*)name
                            gridDim:(MTLSize)gridDim
                           blockDim:(MTLSize)blockDim
                          arguments:(NSArray<id<MTLBuffer>>*)arguments {

                id<MTLComputePipelineState> pipelineState = _kernelPipelineStates[name];
                if (!pipelineState) {
                    return cudaErrorLaunchFailure;
                }

                id<MTLCommandBuffer> commandBuffer = [_commandQueue commandBuffer];
                id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

                // Set compute pipeline state
                [computeEncoder setComputePipelineState:pipelineState];

                // Set buffer arguments
                [arguments enumerateObjectsUsingBlock:^(id<MTLBuffer> buffer, NSUInteger idx, BOOL *stop) {
                    [computeEncoder setBuffer:buffer offset:0 atIndex:idx];
                }];

                // Calculate threadgroup size
                NSUInteger threadGroupWidth = blockDim.width;
                NSUInteger threadGroupHeight = blockDim.height;
                NSUInteger threadGroupDepth = blockDim.depth;

                MTLSize threadsPerThreadgroup = MTLSizeMake(threadGroupWidth,
                                                           threadGroupHeight,
                                                           threadGroupDepth);

                // Dispatch threads
                [computeEncoder dispatchThreadgroups:gridDim
                             threadsPerThreadgroup:threadsPerThreadgroup];

                [computeEncoder endEncoding];
                [commandBuffer commit];

                return cudaSuccess;
            }

            // Helper Methods
            - (CUDAError)setBuffer:(void*)data
                             size:(size_t)size
                        forKernel:(NSString*)kernelName
                           atIndex:(NSUInteger)index {

                id<MTLBuffer> buffer = [_device newBufferWithBytes:data
                                                           length:size
                                                          options:MTLResourceStorageModeShared];
                if (!buffer) {
                    return cudaErrorMemoryAllocation;
                }

                _allocatedBuffers[[NSValue valueWithPointer:buffer.contents]] = buffer;
                return cudaSuccess;
            }

            // CUDA Event Management
            - (CUDAError)cudaEventCreate:(cudaEvent_t*)event {
                *event = (cudaEvent_t)[_device newEvent];
                return cudaSuccess;
            }

            - (CUDAError)cudaEventRecord:(cudaEvent_t)event stream:(cudaStream_t)stream {
                id<MTLCommandBuffer> commandBuffer = (__bridge id<MTLCommandBuffer>)stream;
                [commandBuffer encodeWait:(__bridge id<MTLEvent>)event value:0];
                return cudaSuccess;
            }

            - (CUDAError)cudaEventSynchronize:(cudaEvent_t)event {
                [(id<MTLEvent>)event notifyListener:nil
                                          atValue:0
                                          block:^(id<MTLEvent> event, uint64_t value){}];
                return cudaSuccess;
            }

            // CUDA Stream Management
            - (CUDAError)cudaStreamCreate:(cudaStream_t*)stream {
                *stream = (cudaStream_t)CFBridgingRetain([_commandQueue commandBuffer]);
                return cudaSuccess;
            }

            - (CUDAError)cudaStreamSynchronize:(cudaStream_t)stream {
                id<MTLCommandBuffer> commandBuffer = (__bridge id<MTLCommandBuffer>)stream;
                [commandBuffer waitUntilCompleted];
                return cudaSuccess;
            }

            // Device Synchronization
            - (CUDAError)cudaDeviceSynchronize {
                [_commandQueue insertDebugCaptureBoundary];
                return cudaSuccess;
            }

            @end

            // Kernel Parameters
            @implementation KernelParameters

            - (instancetype)initWithProblemSize:(NSUInteger)problemSize
                                    batchSize:(NSUInteger)batchSize
                               learningRate:(float)learningRate {
                self = [super init];
                if (self) {
                    _problemSize = problemSize;
                    _batchSize = batchSize;
                    _learningRate = learningRate;
                }
                return self;
            }

            - (id<MTLBuffer>)asMetalBufferWithDevice:(id<MTLDevice>)device {
                return [device newBufferWithBytes:self
                                         length:sizeof(KernelParameters)
                                        options:MTLResourceStorageModeShared];
            }

            @end

            // Header file for the above implementation
            @interface CUDAMetalDevice : NSObject

            // CUDA Memory Management
            - (CUDAError)cudaMalloc:(void**)ptr size:(size_t)size;
            - (CUDAError)cudaFree:(void*)ptr;
            - (CUDAError)cudaMemcpy:(void*)dst
                               src:(const void*)src
                              size:(size_t)size
                              kind:(CUDAMemcpyKind)kind;

            // Kernel Management
            - (CUDAError)loadMetalLibraryWithURL:(NSURL*)url error:(NSError**)error;
            - (CUDAError)launchKernel:(NSString*)name
                            gridDim:(MTLSize)gridDim
                           blockDim:(MTLSize)blockDim
                          arguments:(NSArray<id<MTLBuffer>>*)arguments;

            // Event Management
            - (CUDAError)cudaEventCreate:(cudaEvent_t*)event;
            - (CUDAError)cudaEventRecord:(cudaEvent_t)event stream:(cudaStream_t)stream;
            - (CUDAError)cudaEventSynchronize:(cudaEvent_t)event;

            // Stream Management
            - (CUDAError)cudaStreamCreate:(cudaStream_t*)stream;
            - (CUDAError)cudaStreamSynchronize:(cudaStream_t)stream;

            // Device Synchronization
            - (CUDAError)cudaDeviceSynchronize;

            @end