import Metal
import MetalKit

// Entry point for the application using Metal
class MetalApp {
    private let device: MTLDevice
    private let metalManager: MetalManager

    init() {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal is not supported on this device.")
        }
        self.device = device
        self.metalManager = MetalManager(device: device)
    }

    func run() {
        // Input and output buffers setup
        let inputBuffer = device.makeBuffer(length: MemoryLayout<Float>.size * 256, options: [])
        let outputBuffer = device.makeBuffer(length: MemoryLayout<Float>.size * 256, options: [])

        // Fill the input buffer with data
        let inputPointer = inputBuffer?.contents().bindMemory(to: Float.self, capacity: 256)
        for i in 0..<256 {
            inputPointer?[i] = Float(i)
        }

        // Execute kernel
        metalManager.executeKernel(functionName: "example_kernel", inputBuffer: inputBuffer!, outputBuffer: outputBuffer!)
    }
}

// Running the Metal app
let app = MetalApp()
app.run()
