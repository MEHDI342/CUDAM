#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import "metal_manager.h"

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        // Check if Metal is supported
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            NSLog(@"Metal is not supported on this device.");
            return -1;
        }

        // Initialize Metal manager
        MetalManager *metalManager = [[MetalManager alloc] initWithDevice:device];

        // Create input and output buffers
        id<MTLBuffer> inputBuffer = [device newBufferWithLength:sizeof(float) * 256 options:MTLResourceStorageModeShared];
        id<MTLBuffer> outputBuffer = [device newBufferWithLength:sizeof(float) * 256 options:MTLResourceStorageModeShared];

        // Fill input buffer with data
        float *inputPointer = (float *)[inputBuffer contents];
        for (int i = 0; i < 256; i++) {
            inputPointer[i] = (float)i;
        }

        // Execute the kernel
        [metalManager executeKernelWithName:@"example_kernel" withInput:inputBuffer outputBuffer:outputBuffer];

        // Output the results
        float *outputPointer = (float *)[outputBuffer contents];
        for (int i = 0; i < 256; i++) {
            NSLog(@"Output[%d]: %f", i, outputPointer[i]);
        }
    }
    return 0;
}
