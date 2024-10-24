import MetalPerformanceShaders

class CUDNNWrapper {
    private let device: MTLDevice
    private var convolution: MPSCNNConvolution

    init(device: MTLDevice) {
        self.device = device

        let convDesc = MPSCNNConvolutionDescriptor(kernelWidth: 3, kernelHeight: 3,
                                                   inputFeatureChannels: 1, outputFeatureChannels: 1)

        convolution = MPSCNNConvolution(device: device, convolutionDescriptor: convDesc, kernelWeights: [], biasTerms: nil)
    }

    func performConvolution(input: MPSImage, output: MPSImage, commandBuffer: MTLCommandBuffer) {
        convolution.encode(commandBuffer: commandBuffer, sourceImage: input, destinationImage: output)
    }
}
