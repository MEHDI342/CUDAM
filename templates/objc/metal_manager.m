#import "metal_manager.h"

@implementation MetalManager {
    id<MTLDevice> _device;
    id<MTLCommandQueue> _commandQueue;
}

- (instancetype)initWithDevice:(id<MTLDevice>)device {
    self = [super init];
    if (self) {
        _device = device;
        _commandQueue = [_device newCommandQueue];
    }
    return self;
}

- (void)executeKernelWithName:(NSString *)kernelName
                    withInput:(id<MTLBuffer>)inputBuffer
                   outputBuffer:(id<MTLBuffer>)outputBuffer {
    NSError *error = nil;
    id<MTLLibrary> library = [_device newDefaultLibrary];
    id<MTLFunction> function = [library newFunctionWithName:kernelName];

    if (!function) {
        NSLog(@"Failed to load kernel function: %@", kernelName);
        return;
    }

    id<MTLComputePipelineState> pipelineState = [_device newComputePipelineStateWithFunction:function error:&error];
    if (error) {
        NSLog(@"Error creating pipeline state: %@", error.localizedDescription);
        return;
    }

    id<MTLCommandBuffer> commandBuffer = [_commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];

    [commandEncoder setComputePipelineState:pipelineState];
    [commandEncoder setBuffer:inputBuffer offset:0 atIndex:0];
    [commandEncoder setBuffer:outputBuffer offset:0 atIndex:1];

    MTLSize gridSize = MTLSizeMake(256, 1, 1);
    MTLSize threadGroupSize = MTLSizeMake(16, 1, 1);
    [commandEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];

    [commandEncoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    NSLog(@"Kernel execution complete.");
}

@end
