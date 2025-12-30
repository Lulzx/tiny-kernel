const std = @import("std");
const objc = @import("objc.zig");

pub const Device = struct {
    handle: *objc.MTLDevice,
    queue: *objc.MTLCommandQueue,
    pipeline_cache: std.AutoHashMap(u64, *objc.MTLComputePipelineState),
    allocator: std.mem.Allocator,

    pub fn init() !*Device {
        return initWithAllocator(std.heap.page_allocator);
    }

    pub fn initWithAllocator(allocator: std.mem.Allocator) !*Device {
        const device_handle = objc.createSystemDefaultDevice() orelse
            return error.NoMetalDevice;

        const queue = device_handle.newCommandQueue() orelse
            return error.FailedToCreateCommandQueue;

        const self = try allocator.create(Device);
        self.* = .{
            .handle = device_handle,
            .queue = queue,
            .pipeline_cache = std.AutoHashMap(u64, *objc.MTLComputePipelineState).init(allocator),
            .allocator = allocator,
        };
        return self;
    }

    pub fn deinit(self: *Device) void {
        var iter = self.pipeline_cache.valueIterator();
        while (iter.next()) |pipeline| {
            @as(*objc.NSObject, @ptrCast(pipeline.*)).release();
        }
        self.pipeline_cache.deinit();
        @as(*objc.NSObject, @ptrCast(self.queue)).release();
        @as(*objc.NSObject, @ptrCast(self.handle)).release();
        self.allocator.destroy(self);
    }

    pub fn name(self: *Device) []const u8 {
        const ns_name = self.handle.name();
        return std.mem.span(ns_name.UTF8String());
    }

    pub fn compileKernel(self: *Device, source: [:0]const u8, function_name: [:0]const u8) !*objc.MTLComputePipelineState {
        const hash = std.hash.Wyhash.hash(0, source);
        if (self.pipeline_cache.get(hash)) |cached| {
            return cached;
        }

        const ns_source = objc.NSString.initWithUTF8String(source);
        defer @as(*objc.NSObject, @ptrCast(ns_source)).release();

        var err: ?*objc.NSError = null;
        const library = self.handle.newLibraryWithSource(ns_source, null, &err) orelse {
            if (err) |e| {
                const desc = e.localizedDescription();
                std.log.err("Metal compile error: {s}", .{std.mem.span(desc.UTF8String())});
            }
            return error.ShaderCompilationFailed;
        };
        defer @as(*objc.NSObject, @ptrCast(library)).release();

        const ns_func_name = objc.NSString.initWithUTF8String(function_name);
        defer @as(*objc.NSObject, @ptrCast(ns_func_name)).release();

        const function = library.newFunctionWithName(ns_func_name) orelse
            return error.FunctionNotFound;
        defer @as(*objc.NSObject, @ptrCast(function)).release();

        const pipeline = self.handle.newComputePipelineStateWithFunction(function, &err) orelse {
            if (err) |e| {
                const desc = e.localizedDescription();
                std.log.err("Pipeline creation error: {s}", .{std.mem.span(desc.UTF8String())});
            }
            return error.PipelineCreationFailed;
        };

        try self.pipeline_cache.put(hash, pipeline);
        return pipeline;
    }

    pub fn createBuffer(self: *Device, comptime T: type, count: usize) !Buffer(T) {
        return Buffer(T).init(self.handle, count);
    }

    pub fn createBufferWithData(self: *Device, comptime T: type, data: []const T) !Buffer(T) {
        return Buffer(T).initWithData(self.handle, data);
    }

    pub fn createCommandBuffer(self: *Device) !CommandBuffer {
        const cmd_buffer = self.queue.commandBuffer() orelse
            return error.FailedToCreateCommandBuffer;
        return CommandBuffer{ .handle = cmd_buffer };
    }
};

pub fn Buffer(comptime T: type) type {
    return struct {
        handle: *objc.MTLBuffer,
        count: usize,

        const Self = @This();

        pub fn init(device: *objc.MTLDevice, count: usize) !Self {
            const size = count * @sizeOf(T);
            const handle = device.newBufferWithLength(size, objc.MTLResourceStorageModeShared) orelse
                return error.FailedToCreateBuffer;
            return .{ .handle = handle, .count = count };
        }

        pub fn initWithData(device: *objc.MTLDevice, data: []const T) !Self {
            const size = data.len * @sizeOf(T);
            const handle = device.newBufferWithBytes(data.ptr, size, objc.MTLResourceStorageModeShared) orelse
                return error.FailedToCreateBuffer;
            return .{ .handle = handle, .count = data.len };
        }

        pub fn deinit(self: *Self) void {
            @as(*objc.NSObject, @ptrCast(self.handle)).release();
        }

        pub fn slice(self: *Self) []T {
            const ptr: [*]T = @ptrCast(@alignCast(self.handle.contents()));
            return ptr[0..self.count];
        }

        pub fn constSlice(self: *const Self) []const T {
            const ptr: [*]const T = @ptrCast(@alignCast(self.handle.contents()));
            return ptr[0..self.count];
        }
    };
}

pub const CommandBuffer = struct {
    handle: *objc.MTLCommandBuffer,

    pub fn computeEncoder(self: *CommandBuffer) !ComputeEncoder {
        const encoder = self.handle.computeCommandEncoder() orelse
            return error.FailedToCreateEncoder;
        return .{ .handle = encoder };
    }

    pub fn commit(self: *CommandBuffer) void {
        self.handle.commit();
    }

    pub fn waitUntilCompleted(self: *CommandBuffer) void {
        self.handle.waitUntilCompleted();
    }
};

pub const ComputeEncoder = struct {
    handle: *objc.MTLComputeCommandEncoder,

    pub fn setPipeline(self: *ComputeEncoder, pipeline: *objc.MTLComputePipelineState) void {
        self.handle.setComputePipelineState(pipeline);
    }

    pub fn setBuffer(self: *ComputeEncoder, comptime T: type, buffer: *Buffer(T), index: usize) void {
        self.handle.setBuffer(buffer.handle, 0, index);
    }

    pub fn setBytes(self: *ComputeEncoder, comptime T: type, value: *const T, index: usize) void {
        self.handle.setBytes(value, @sizeOf(T), index);
    }

    pub fn dispatch(self: *ComputeEncoder, grid: objc.MTLSize, threadgroup: objc.MTLSize) void {
        self.handle.dispatchThreadgroups(grid, threadgroup);
    }

    pub fn dispatchThreads(self: *ComputeEncoder, threads: objc.MTLSize, threadgroup: objc.MTLSize) void {
        self.handle.dispatchThreads(threads, threadgroup);
    }

    pub fn endEncoding(self: *ComputeEncoder) void {
        self.handle.endEncoding();
    }
};
