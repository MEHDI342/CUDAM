import Metal
import Foundation

class MetalManager {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue

    init(device: MTLDevice) {
        self.device = device
        self.commandQueue = device.makeCommandQueue()!
    }

    func executeKernel(functionName: String, inputBuffer: MTLBuffer, outputBuffer: MTLBuffer) {
        guard let library = device.makeDefaultLibrary(),
              let function = library.makeFunction(name: functionName) else {
            print("Failed to find the function \(functionName)")
            return
        }

        do {
            let pipelineState = try device.makeComputePipelineState(function: function)
            guard let commandBuffer = commandQueue.makeCommandBuffer(),
                  let commandEncoder = commandBuffer.makeComputeCommandEncoder() else {
                print("Failed to create command encoder")
                return
            }

            commandEncoder.setComputePipelineState(pipelineState)
            commandEncoder.setBuffer(inputBuffer, offset: 0, index: 0)
            commandEncoder.setBuffer(outputBuffer, offset: 0, index: 1)

            let gridSize = MTLSize(width: 256, height: 1, depth: 1)
            let threadGroupSize = MTLSize(width: 16, height: 1, depth: 1)
            commandEncoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)

            commandEncoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()

            print("Kernel execution completed")
        } catch {
            print("Error creating pipeline state: \(error)")
        }
    }
}
