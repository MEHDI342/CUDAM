#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

@interface MetalManager : NSObject

- (instancetype)initWithDevice:(id<MTLDevice>)device;
- (void)executeKernelWithName:(NSString *)kernelName
                    withInput:(id<MTLBuffer>)inputBuffer
                   outputBuffer:(id<MTLBuffer>)outputBuffer;

@end
