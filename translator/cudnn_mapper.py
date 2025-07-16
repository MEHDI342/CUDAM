from typing import Dict, List, Any
from ..utils.error_handler import CudaTranslationError
from ..utils.logger import get_logger

logger = get_logger(__name__)

class CudnnMapper:
    def __init__(self):
        self.cudnn_to_mps_map: Dict[str, str] = {
            'cudnnConvolutionForward': 'MPSCNNConvolution',
            'cudnnPoolingForward': 'MPSCNNPooling',
            'cudnnActivationForward': 'MPSCNNNeuron',
            'cudnnSoftmaxForward': 'MPSCNNSoftMax',
            'cudnnBatchNormalizationForward': 'MPSCNNBatchNormalization',
            'cudnnRNNForward': 'MPSNNGRU',
            'cudnnDropoutForward': 'MPSCNNDropout',
            'cudnnOpTensor': 'MPSNNAdd',
        }

    def map_function(self, cudnn_function: str, args: List[Any]) -> str:
        if cudnn_function not in self.cudnn_to_mps_map:
            raise CudaTranslationError(f"Unsupported cuDNN function: {cudnn_function}")

        mps_function = self.cudnn_to_mps_map[cudnn_function]
        return self._generate_mps_call(mps_function, args)

    def _generate_mps_call(self, mps_function: str, args: List[Any]) -> str:
        if mps_function == 'MPSCNNConvolution':
            return self._generate_convolution_call(args)
        elif mps_function == 'MPSCNNPooling':
            return self._generate_pooling_call(args)
        elif mps_function == 'MPSCNNNeuron':
            return self._generate_activation_call(args)
        elif mps_function == 'MPSCNNSoftMax':
            return self._generate_softmax_call(args)
        elif mps_function == 'MPSCNNBatchNormalization':
            return self._generate_batchnorm_call(args)
        else:
            return f"{mps_function}({', '.join(map(str, args))})"

    def _generate_convolution_call(self, args: List[Any]) -> str:
        return f"""
        MPSCNNConvolution *convLayer = [[MPSCNNConvolution alloc]
            initWithDevice:device
            kernelWidth:{args[0]}
            kernelHeight:{args[1]}
            inputFeatureChannels:{args[2]}
            outputFeatureChannels:{args[3]}
            neuronFilter:nil];
        [convLayer encodeToCommandBuffer:commandBuffer
            sourceImage:sourceTexture
            destinationImage:destTexture];
        """

    def _generate_pooling_call(self, args: List[Any]) -> str:
        return f"""
        MPSCNNPooling *poolLayer = [[MPSCNNPooling alloc]
            initWithDevice:device
            kernelWidth:{args[0]}
            kernelHeight:{args[1]}
            strideInPixelsX:{args[2]}
            strideInPixelsY:{args[3]}];
        [poolLayer encodeToCommandBuffer:commandBuffer
            sourceImage:sourceTexture
            destinationImage:destTexture];
        """

    def _generate_activation_call(self, args: List[Any]) -> str:
        return f"""
        MPSCNNNeuron *activationLayer = [MPSCNNNeuronReLU nodeWithSource:nil];
        [activationLayer encodeToCommandBuffer:commandBuffer
            sourceImage:sourceTexture
            destinationImage:destTexture];
        """

    def _generate_softmax_call(self, args: List[Any]) -> str:
        return f"""
        MPSCNNSoftMax *softmaxLayer = [[MPSCNNSoftMax alloc] initWithDevice:device];
        [softmaxLayer encodeToCommandBuffer:commandBuffer
            sourceImage:sourceTexture
            destinationImage:destTexture];
        """

    def _generate_batchnorm_call(self, args: List[Any]) -> str:
        return f"""
        MPSCNNBatchNormalization *batchNormLayer = [[MPSCNNBatchNormalization alloc]
            initWithDevice:device
            featureChannels:{args[0]}];
        [batchNormLayer encodeToCommandBuffer:commandBuffer
            sourceImage:sourceTexture
            destinationImage:destTexture];
        """

    def translate_cudnn_descriptor(self, descriptor_type: str, params: Dict[str, Any]) -> str:
        if descriptor_type == 'cudnnTensorDescriptor':
            return self._translate_tensor_descriptor(params)
        elif descriptor_type == 'cudnnFilterDescriptor':
            return self._translate_filter_descriptor(params)
        elif descriptor_type == 'cudnnConvolutionDescriptor':
            return self._translate_convolution_descriptor(params)
        else:
            raise CudaTranslationError(f"Unsupported descriptor type: {descriptor_type}")

    def _translate_tensor_descriptor(self, params: Dict[str, Any]) -> str:
        return f"""
        MPSImageDescriptor *tensorDescriptor = [MPSImageDescriptor
            imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat32
            width:{params['width']}
            height:{params['height']}
            featureChannels:{params['channels']}];
        """

    def _translate_filter_descriptor(self, params: Dict[str, Any]) -> str:
        return f"""
        MPSCNNConvolutionDescriptor *filterDescriptor = [MPSCNNConvolutionDescriptor
            cnnConvolutionDescriptorWithKernelWidth:{params['kernelWidth']}
            kernelHeight:{params['kernelHeight']}
            inputFeatureChannels:{params['inputChannels']}
            outputFeatureChannels:{params['outputChannels']}];
        """

    def _translate_convolution_descriptor(self, params: Dict[str, Any]) -> str:
        return f"""
        MPSNNDefaultPadding *convolutionDescriptor = [MPSNNDefaultPadding
            paddingWithMethod:MPSNNPaddingMethodSizeSame];
        convolutionDescriptor.kernelOffsetX = {params['padWidth']};
        convolutionDescriptor.kernelOffsetY = {params['padHeight']};
        """

logger.info("CudnnMapper initialized for cuDNN to Metal Performance Shaders translation.")