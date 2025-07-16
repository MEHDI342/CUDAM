#import <Foundation/Foundation.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

@interface CUDNNWrapper : NSObject

- (instancetype)initWithDevice:(id<MTLDevice>)device;
- (void)performConvolutionWithInput:(MPSImage *)input
                             output:(MPSImage *)output;

@end
