#import "cudnn_wrapper.h"

@implementation CUDNNWrapper {
    id<MTLDevice> _device;
    MPSNNConvolution *convolution;
}

- (instancetype)initWithDevice:(id<MTLDevice>)device {
    self = [super init];
    if (self) {
        _device = device;
        // Setup Metal Performance Shader convolution kernel
        MPSNNConvolutionDescriptor *convDesc = [[MPSNNConvolutionDescriptor alloc] initWithKernelWidth:3
                                                                                          kernelHeight:3
                                                                                      inputFeatureChannels:1
                                                                                     outputFeatureChannels:1];
        convolution = [[MPSNNConvolution alloc] initWithDevice:_device
                                              convolutionDescriptor:convDesc];
    }
    return self;
}

- (void)performConvolutionWithInput:(MPSImage *)input
                             output:(MPSImage *)output {
    // Code to perform convolution
    // Example only: Ensure input/output handling is correct in actual code
    [convolution encodeToCommandBuffer:commandBuffer
                                sourceImage:input
                           destinationImage:output];
}

@end
