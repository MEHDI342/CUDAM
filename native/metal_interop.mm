// metal_interop.mm
// (Continuing the implementation of all remaining functions)

void begin_compute_pass(MetalCommandObjects* cmd_objects) {
    if (!cmd_objects || cmd_objects->compute_encoder) return;

    id<MTLCommandBuffer> cmdBuffer = (__bridge id<MTLCommandBuffer>)cmd_objects->command_buffer;
    id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];
    cmd_objects->compute_encoder = (__bridge_retained void*)encoder;
}

void end_compute_pass(MetalCommandObjects* cmd_objects) {
    if (!cmd_objects || !cmd_objects->compute_encoder) return;

    id<MTLComputeCommandEncoder> encoder = (__bridge id<MTLComputeCommandEncoder>)cmd_objects->compute_encoder;
    [encoder endEncoding];

    cmd_objects->compute_encoder = nil;
}

void commit_commands(MetalCommandObjects* cmd_objects) {
    if (!cmd_objects || !cmd_objects->command_buffer) return;

    id<MTLCommandBuffer> cmdBuffer = (__bridge id<MTLCommandBuffer>)cmd_objects->command_buffer;
    [cmdBuffer commit];
}

void wait_for_completion(MetalCommandObjects* cmd_objects) {
    if (!cmd_objects || !cmd_objects->command_buffer) return;

    id<MTLCommandBuffer> cmdBuffer = (__bridge id<MTLCommandBuffer>)cmd_objects->command_buffer;
    [cmdBuffer waitUntilCompleted];
}

MetalPipelineConfig* create_pipeline_config(const char* kernel_name) {
    if (!kernel_name) return NULL;

    MetalPipelineConfig* config = (MetalPipelineConfig*)malloc(sizeof(MetalPipelineConfig));
    if (!config) return NULL;

    NSString* funcName = [NSString stringWithUTF8String:kernel_name];
    id<MTLDevice> device = [MetalDeviceManager sharedDevice];
    id<MTLLibrary> library = [device newDefaultLibrary];
    id<MTLFunction> function = [library newFunctionWithName:funcName];

    NSError* error = nil;
    id<MTLComputePipelineState> pipelineState =
        [device newComputePipelineStateWithFunction:function error:&error];

    if (!pipelineState) {
        NSLog(@"Failed to create pipeline state: %@", error);
        free(config);
        return NULL;
    }

    config->pipeline_state = (__bridge_retained void*)pipelineState;
    config->thread_group_size[0] = 1;
    config->thread_group_size[1] = 1;
    config->thread_group_size[2] = 1;
    config->grid_size[0] = 1;
    config->grid_size[1] = 1;
    config->grid_size[2] = 1;

    return config;
}

void destroy_pipeline_config(MetalPipelineConfig* config) {
    if (!config) return;

    if (config->pipeline_state) {
        id<MTLComputePipelineState> pipelineState =
            (__bridge_transfer id<MTLComputePipelineState>)config->pipeline_state;
        pipelineState = nil;
    }

    free(config);
}

void set_pipeline_thread_groups(MetalPipelineConfig* config,
                              uint32_t x, uint32_t y, uint32_t z) {
    if (!config) return;

    config->thread_group_size[0] = x;
    config->thread_group_size[1] = y;
    config->thread_group_size[2] = z;
}

void set_pipeline_grid_size(MetalPipelineConfig* config,
                           uint32_t x, uint32_t y, uint32_t z) {
    if (!config) return;

    config->grid_size[0] = x;
    config->grid_size[1] = y;
    config->grid_size[2] = z;
}

@interface MetalCommandBufferWrapper : NSObject
@property (nonatomic, strong) id<MTLCommandBuffer> commandBuffer;
@property (nonatomic, strong) NSMutableArray<id<MTLBuffer>>* retainedBuffers;
@end

@implementation MetalCommandBufferWrapper
- (instancetype)initWithCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    if (self = [super init]) {
        _commandBuffer = commandBuffer;
        _retainedBuffers = [NSMutableArray array];
    }
    return self;
}
@end

// Thread-local storage for retained buffers
static NSMutableDictionary* threadLocalBuffers = nil;
static dispatch_once_t bufferOnceToken;

@interface MetalBufferManager : NSObject
+ (void)retainBuffer:(id<MTLBuffer>)buffer forThread:(NSThread*)thread;
+ (void)releaseBuffersForThread:(NSThread*)thread;
@end

@implementation MetalBufferManager

+ (void)initialize {
    if (self == [MetalBufferManager class]) {
        dispatch_once(&bufferOnceToken, ^{
            threadLocalBuffers = [NSMutableDictionary dictionary];
        });
    }
}

+ (void)retainBuffer:(id<MTLBuffer>)buffer forThread:(NSThread*)thread {
    if (!buffer || !thread) return;

    @synchronized(threadLocalBuffers) {
        NSString* threadKey = [NSString stringWithFormat:@"%p", thread];
        NSMutableArray* buffers = threadLocalBuffers[threadKey];
        if (!buffers) {
            buffers = [NSMutableArray array];
            threadLocalBuffers[threadKey] = buffers;
        }
        [buffers addObject:buffer];
    }
}

+ (void)releaseBuffersForThread:(NSThread*)thread {
    if (!thread) return;

    @synchronized(threadLocalBuffers) {
        NSString* threadKey = [NSString stringWithFormat:@"%p", thread];
        [threadLocalBuffers removeObjectForKey:threadKey];
    }
}

@end

// Helper functions for error handling
static void handleMetalError(NSError* error, const char* operation) {
    if (error) {
        NSLog(@"Metal error during %s: %@", operation, error);
    }
}

static BOOL validateDevice() {
    id<MTLDevice> device = [MetalDeviceManager sharedDevice];
    if (!device) {
        NSLog(@"No Metal device available");
        return NO;
    }
    return YES;
}

static BOOL validatePipelineState(id<MTLComputePipelineState> pipelineState) {
    if (!pipelineState) {
        NSLog(@"Invalid compute pipeline state");
        return NO;
    }
    return YES;
}