import Metal
import Foundation

// Advanced error handling for Metal operations
public enum MetalError: Error {
    case deviceNotFound
    case libraryCreationFailed
    case commandCreationFailed
    case pipelineCreationFailed
    case bufferCreationFailed
    case invalidThreadgroupSize
    case computeFailure(String)
    case resourceAllocationFailure
    case invalidKernelName
    case unsupportedOperation
}

// Protocol for Metal kernel execution
public protocol MetalKernelExecutable {
    func executeKernel(name: String,
                      buffers: [MTLBuffer],
                      threadgroupSize: MTLSize,
                      gridSize: MTLSize) throws

    func executeKernelAsync(name: String,
                           buffers: [MTLBuffer],
                           threadgroupSize: MTLSize,
                           gridSize: MTLSize,
                           completion: @escaping (Error?) -> Void)
}

// Main Metal kernel executor implementation
public final class MetalKernelExecutor: MetalKernelExecutable {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let pipelineCache: NSCache<NSString, MTLComputePipelineState>
    private let resourceSemaphore: DispatchSemaphore
    private let executionQueue: DispatchQueue

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalError.deviceNotFound
        }

        guard let commandQueue = device.makeCommandQueue() else {
            throw MetalError.commandCreationFailed
        }

        self.device = device
        self.commandQueue = commandQueue
        self.pipelineCache = NSCache()
        self.resourceSemaphore = DispatchSemaphore(value: 3) // Limit concurrent executions
        self.executionQueue = DispatchQueue(label: "com.metal.execution",
                                          qos: .userInitiated,
                                          attributes: .concurrent)

        // Configure cache limits
        pipelineCache.countLimit = 50
    }

    public func executeKernel(
        name: String,
        buffers: [MTLBuffer],
        threadgroupSize: MTLSize,
        gridSize: MTLSize
    ) throws {
        // Validate inputs
        guard !name.isEmpty else {
            throw MetalError.invalidKernelName
        }

        guard isValidThreadgroupSize(threadgroupSize) else {
            throw MetalError.invalidThreadgroupSize
        }

        // Wait for available resource slot
        resourceSemaphore.wait()

        defer {
            resourceSemaphore.signal()
        }

        do {
            // Get pipeline state
            let pipelineState = try getPipelineState(kernelName: name)

            // Create command buffer and encoder
            guard let commandBuffer = commandQueue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw MetalError.commandCreationFailed
            }

            // Configure compute encoder
            encoder.setComputePipelineState(pipelineState)

            // Bind buffers
            for (index, buffer) in buffers.enumerated() {
                encoder.setBuffer(buffer, offset: 0, index: index)
            }

            // Validate and adjust sizes
            let adjustedSizes = calculateOptimalSizes(
                pipeline: pipelineState,
                requestedThreadgroup: threadgroupSize,
                requestedGrid: gridSize
            )

            // Dispatch compute kernel
            encoder.dispatchThreadgroups(adjustedSizes.grid,
                                       threadsPerThreadgroup: adjustedSizes.threadgroup)

            // Complete encoding and commit
            encoder.endEncoding()
            commandBuffer.commit()

            // Wait for completion and handle errors
            commandBuffer.waitUntilCompleted()

            if let error = commandBuffer.error {
                throw MetalError.computeFailure(error.localizedDescription)
            }

        } catch {
            throw MetalError.computeFailure("Kernel execution failed: \(error.localizedDescription)")
        }
    }

    public func executeKernelAsync(
        name: String,
        buffers: [MTLBuffer],
        threadgroupSize: MTLSize,
        gridSize: MTLSize,
        completion: @escaping (Error?) -> Void
    ) {
        executionQueue.async { [weak self] in
            do {
                try self?.executeKernel(
                    name: name,
                    buffers: buffers,
                    threadgroupSize: threadgroupSize,
                    gridSize: gridSize
                )
                completion(nil)
            } catch {
                completion(error)
            }
        }
    }

    private func getPipelineState(kernelName: String) throws -> MTLComputePipelineState {
        let key = kernelName as NSString

        // Check cache
        if let cached = pipelineCache.object(forKey: key) {
            return cached
        }

        // Create new pipeline state
        guard let library = device.makeDefaultLibrary(),
              let function = library.makeFunction(name: kernelName) else {
            throw MetalError.libraryCreationFailed
        }

        let descriptor = MTLComputePipelineDescriptor()
        descriptor.computeFunction = function
        descriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth = true

        let options: MTLPipelineOption = [.argumentInfo, .bufferTypeInfo]

        let pipelineState = try device.makeComputePipelineState(
            descriptor: descriptor,
            options: options,
            reflection: nil
        )

        pipelineCache.setObject(pipelineState, forKey: key)
        return pipelineState
    }

    private func isValidThreadgroupSize(_ size: MTLSize) -> Bool {
        let maxTotal = device.maxThreadsPerThreadgroup
        let total = size.width * size.height * size.depth
        return total <= maxTotal
    }

    private func calculateOptimalSizes(
        pipeline: MTLComputePipelineState,
        requestedThreadgroup: MTLSize,
        requestedGrid: MTLSize
    ) -> (threadgroup: MTLSize, grid: MTLSize) {
        // Get optimal thread execution width
        let width = pipeline.threadExecutionWidth
        let height = pipeline.maxTotalThreadsPerThreadgroup / width

        // Adjust threadgroup size
        let threadgroup = MTLSize(
            width: min(requestedThreadgroup.width, width),
            height: min(requestedThreadgroup.height, height),
            depth: 1
        )

        // Calculate grid size
        let grid = MTLSize(
            width: (requestedGrid.width + threadgroup.width - 1) / threadgroup.width,
            height: (requestedGrid.height + threadgroup.height - 1) / threadgroup.height,
            depth: requestedGrid.depth
        )

        return (threadgroup, grid)
    }
}

// Resource manager for Metal buffers and textures
public final class MetalResourceManager {
    private let device: MTLDevice
    private var bufferCache: [String: WeakBuffer] = [:]
    private let queue = DispatchQueue(label: "com.metal.resourcemanager")
    private let maxBufferSize: Int

    private class WeakBuffer {
        weak var buffer: MTLBuffer?
        let creationTime: Date

        init(_ buffer: MTLBuffer) {
            self.buffer = buffer
            self.creationTime = Date()
        }
    }

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalError.deviceNotFound
        }
        self.device = device
        self.maxBufferSize = device.maxBufferLength

        // Start cache cleanup timer
        startCacheCleanupTimer()
    }

    public func createBuffer(
        size: Int,
        options: MTLResourceOptions = []
    ) throws -> MTLBuffer {
        guard size > 0 && size <= maxBufferSize else {
            throw MetalError.bufferCreationFailed
        }

        guard let buffer = device.makeBuffer(length: size, options: options) else {
            throw MetalError.bufferCreationFailed
        }

        return buffer
    }

    public func getOrCreateBuffer(
            identifier: String,
            size: Int,
            options: MTLResourceOptions = []
        ) throws -> MTLBuffer {
            return try queue.sync {
                // Clean up expired cache entries
                cleanupExpiredBuffers()

                // Check cache for existing buffer
                if let weakBuffer = bufferCache[identifier],
                   let buffer = weakBuffer.buffer,
                   buffer.length >= size {
                    return buffer
                }

                // Create new buffer
                let buffer = try createBuffer(size: size, options: options)
                bufferCache[identifier] = WeakBuffer(buffer)
                return buffer
            }
        }

        public func clearCache() {
            queue.sync {
                bufferCache.removeAll()
            }
        }

        private func cleanupExpiredBuffers() {
            let now = Date()
            bufferCache = bufferCache.filter { identifier, weakBuffer in
                guard let _ = weakBuffer.buffer else { return false }
                // Keep buffers that are less than 5 minutes old
                return now.timeIntervalSince(weakBuffer.creationTime) < 300
            }
        }

        private func startCacheCleanupTimer() {
            Timer.scheduledTimer(withTimeInterval: 60, repeats: true) { [weak self] _ in
                self?.queue.async {
                    self?.cleanupExpiredBuffers()
                }
            }
        }

        // Advanced buffer management methods
        public func copyBuffer(_ sourceBuffer: MTLBuffer,
                             to destinationBuffer: MTLBuffer,
                             size: Int) throws {
            guard size <= sourceBuffer.length && size <= destinationBuffer.length else {
                throw MetalError.bufferCreationFailed
            }

            guard let commandBuffer = device.makeCommandQueue()?.makeCommandBuffer(),
                  let blitEncoder = commandBuffer.makeBlitCommandEncoder() else {
                throw MetalError.commandCreationFailed
            }

            blitEncoder.copy(from: sourceBuffer,
                            sourceOffset: 0,
                            to: destinationBuffer,
                            destinationOffset: 0,
                            size: size)

            blitEncoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }

        public func fillBuffer(_ buffer: MTLBuffer,
                             with value: UInt8,
                             range: Range<Int>? = nil) throws {
            let fillRange = range ?? 0..<buffer.length

            guard let commandBuffer = device.makeCommandQueue()?.makeCommandBuffer(),
                  let blitEncoder = commandBuffer.makeBlitCommandEncoder() else {
                throw MetalError.commandCreationFailed
            }

            blitEncoder.fill(buffer: buffer,
                            range: fillRange,
                            value: value)

            blitEncoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }

        // Texture management
        public func createTexture(
            width: Int,
            height: Int,
            pixelFormat: MTLPixelFormat,
            usage: MTLTextureUsage = [.shaderRead, .shaderWrite]
        ) throws -> MTLTexture {
            let descriptor = MTLTextureDescriptor()
            descriptor.textureType = .type2D
            descriptor.width = width
            descriptor.height = height
            descriptor.pixelFormat = pixelFormat
            descriptor.usage = usage

            guard let texture = device.makeTexture(descriptor: descriptor) else {
                throw MetalError.resourceAllocationFailure
            }

            return texture
        }

        // Buffer synchronization
        public func synchronizeBuffer(_ buffer: MTLBuffer) throws {
            guard let commandBuffer = device.makeCommandQueue()?.makeCommandBuffer(),
                  let blitEncoder = commandBuffer.makeBlitCommandEncoder() else {
                throw MetalError.commandCreationFailed
            }

            blitEncoder.synchronize(resource: buffer)
            blitEncoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }

        // Memory management helpers
        public func purgeableState(for buffer: MTLBuffer) -> MTLPurgeableState {
            return buffer.setPurgeableState(.empty)
        }

        public func makeBufferPurgeable(_ buffer: MTLBuffer) {
            _ = buffer.setPurgeableState(.volatile)
        }

        public func makeBufferNonPurgeable(_ buffer: MTLBuffer) {
            _ = buffer.setPurgeableState(.nonVolatile)
        }

        // Memory statistics
        public func getMemoryStats() -> (used: Int, total: Int) {
            var used = 0
            queue.sync {
                for (_, weakBuffer) in bufferCache {
                    if let buffer = weakBuffer.buffer {
                        used += buffer.length
                    }
                }
            }
            return (used, maxBufferSize)
        }

        // Resource barriers
        public func deviceMemoryBarrier() throws {
            guard let commandBuffer = device.makeCommandQueue()?.makeCommandBuffer(),
                  let blitEncoder = commandBuffer.makeBlitCommandEncoder() else {
                throw MetalError.commandCreationFailed
            }

            blitEncoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }
    }

    // Extension for convenience methods
    extension MetalResourceManager {
        public func withMappedBuffer<T>(
            _ buffer: MTLBuffer,
            type: T.Type,
            body: (UnsafeMutableBufferPointer<T>) throws -> Void
        ) throws {
            guard let contents = buffer.contents().bindMemory(
                to: type,
                capacity: buffer.length / MemoryLayout<T>.stride
            ) else {
                throw MetalError.resourceAllocationFailure
            }

            let bufferPointer = UnsafeMutableBufferPointer(
                start: contents,
                count: buffer.length / MemoryLayout<T>.stride
            )

            try body(bufferPointer)
        }

        public func createTypedBuffer<T>(
            _ type: T.Type,
            count: Int,
            options: MTLResourceOptions = []
        ) throws -> MTLBuffer {
            let size = count * MemoryLayout<T>.stride
            return try createBuffer(size: size, options: options)
        }
    }

    // Utility extensions for Metal types
    extension MTLSize {
        public static func make(_ width: Int, _ height: Int = 1, _ depth: Int = 1) -> MTLSize {
            return MTLSizeMake(width, height, depth)
        }

        public var total: Int {
            return width * height * depth
        }
    }

    extension MTLBuffer {
        public func contents<T>(as type: T.Type) -> UnsafeMutablePointer<T> {
            return contents().assumingMemoryBound(to: type)
        }
    }