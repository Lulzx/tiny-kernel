const std = @import("std");

pub const id = *opaque {};
pub const SEL = *opaque {};
pub const Class = *opaque {};
pub const IMP = *const fn () callconv(.C) void;

pub const NSObject = opaque {
    pub fn release(self: *NSObject) void {
        _ = msgSend(self, "release", .{}, void);
    }

    pub fn retain(self: *NSObject) *NSObject {
        return msgSend(self, "retain", .{}, *NSObject);
    }
};

extern "c" fn objc_getClass(name: [*:0]const u8) ?Class;
extern "c" fn sel_registerName(name: [*:0]const u8) SEL;
extern "c" fn objc_msgSend() void;

pub fn msgSend(obj: anytype, sel_name: [:0]const u8, args: anytype, comptime ReturnType: type) ReturnType {
    const sel = sel_registerName(sel_name.ptr);
    const FnType = MsgSendFn(@TypeOf(obj), @TypeOf(args), ReturnType);
    const func: *const FnType = @ptrCast(&objc_msgSend);
    return @call(.auto, func, .{obj, sel} ++ args);
}

pub fn classMsgSend(class_name: [:0]const u8, sel_name: [:0]const u8, args: anytype, comptime ReturnType: type) ReturnType {
    const class = objc_getClass(class_name.ptr) orelse @panic("Class not found");
    return msgSend(class, sel_name, args, ReturnType);
}

fn MsgSendFn(comptime ObjType: type, comptime ArgsType: type, comptime ReturnType: type) type {
    const args_info = @typeInfo(ArgsType);
    if (args_info != .@"struct") @compileError("Args must be a tuple");

    const fields = args_info.@"struct".fields;
    var param_types: [fields.len + 2]type = undefined;
    param_types[0] = ObjType;
    param_types[1] = SEL;

    inline for (fields, 0..) |field, i| {
        param_types[i + 2] = field.type;
    }

    return @Type(.{
        .@"fn" = .{
            .calling_convention = std.builtin.CallingConvention.c,
            .is_generic = false,
            .is_var_args = false,
            .return_type = ReturnType,
            .params = blk: {
                var params: [param_types.len]std.builtin.Type.Fn.Param = undefined;
                for (param_types, 0..) |T, i| {
                    params[i] = .{ .is_generic = false, .is_noalias = false, .type = T };
                }
                break :blk &params;
            },
        },
    });
}

pub const MTLDevice = opaque {
    pub fn name(self: *MTLDevice) *NSString {
        return msgSend(self, "name", .{}, *NSString);
    }

    pub fn newCommandQueue(self: *MTLDevice) ?*MTLCommandQueue {
        return msgSend(self, "newCommandQueue", .{}, ?*MTLCommandQueue);
    }

    pub fn newBufferWithLength(self: *MTLDevice, length: usize, options: MTLResourceOptions) ?*MTLBuffer {
        return msgSend(self, "newBufferWithLength:options:", .{ length, options }, ?*MTLBuffer);
    }

    pub fn newBufferWithBytes(self: *MTLDevice, ptr: *const anyopaque, length: usize, options: MTLResourceOptions) ?*MTLBuffer {
        return msgSend(self, "newBufferWithBytes:length:options:", .{ ptr, length, options }, ?*MTLBuffer);
    }

    pub fn newLibraryWithSource(self: *MTLDevice, source: *NSString, options: ?*anyopaque, err: *?*NSError) ?*MTLLibrary {
        return msgSend(self, "newLibraryWithSource:options:error:", .{ source, options, err }, ?*MTLLibrary);
    }

    pub fn newComputePipelineStateWithFunction(self: *MTLDevice, func: *MTLFunction, err: *?*NSError) ?*MTLComputePipelineState {
        return msgSend(self, "newComputePipelineStateWithFunction:error:", .{ func, err }, ?*MTLComputePipelineState);
    }
};

pub const MTLCommandQueue = opaque {
    pub fn commandBuffer(self: *MTLCommandQueue) ?*MTLCommandBuffer {
        return msgSend(self, "commandBuffer", .{}, ?*MTLCommandBuffer);
    }
};

pub const MTLCommandBuffer = opaque {
    pub fn computeCommandEncoder(self: *MTLCommandBuffer) ?*MTLComputeCommandEncoder {
        return msgSend(self, "computeCommandEncoder", .{}, ?*MTLComputeCommandEncoder);
    }

    pub fn commit(self: *MTLCommandBuffer) void {
        _ = msgSend(self, "commit", .{}, void);
    }

    pub fn waitUntilCompleted(self: *MTLCommandBuffer) void {
        _ = msgSend(self, "waitUntilCompleted", .{}, void);
    }
};

pub const MTLComputeCommandEncoder = opaque {
    pub fn setComputePipelineState(self: *MTLComputeCommandEncoder, state: *MTLComputePipelineState) void {
        _ = msgSend(self, "setComputePipelineState:", .{state}, void);
    }

    pub fn setBuffer(self: *MTLComputeCommandEncoder, buffer: *MTLBuffer, offset: usize, index: usize) void {
        _ = msgSend(self, "setBuffer:offset:atIndex:", .{ buffer, offset, index }, void);
    }

    pub fn setBytes(self: *MTLComputeCommandEncoder, bytes: *const anyopaque, length: usize, index: usize) void {
        _ = msgSend(self, "setBytes:length:atIndex:", .{ bytes, length, index }, void);
    }

    pub fn dispatchThreadgroups(self: *MTLComputeCommandEncoder, threadgroups: MTLSize, threads_per_threadgroup: MTLSize) void {
        _ = msgSend(self, "dispatchThreadgroups:threadsPerThreadgroup:", .{ threadgroups, threads_per_threadgroup }, void);
    }

    pub fn dispatchThreads(self: *MTLComputeCommandEncoder, threads: MTLSize, threads_per_threadgroup: MTLSize) void {
        _ = msgSend(self, "dispatchThreads:threadsPerThreadgroup:", .{ threads, threads_per_threadgroup }, void);
    }

    pub fn endEncoding(self: *MTLComputeCommandEncoder) void {
        _ = msgSend(self, "endEncoding", .{}, void);
    }
};

pub const MTLBuffer = opaque {
    pub fn contents(self: *MTLBuffer) *anyopaque {
        return msgSend(self, "contents", .{}, *anyopaque);
    }

    pub fn length(self: *MTLBuffer) usize {
        return msgSend(self, "length", .{}, usize);
    }
};

pub const MTLLibrary = opaque {
    pub fn newFunctionWithName(self: *MTLLibrary, n: *NSString) ?*MTLFunction {
        return msgSend(self, "newFunctionWithName:", .{n}, ?*MTLFunction);
    }
};

pub const MTLFunction = opaque {};
pub const MTLComputePipelineState = opaque {
    pub fn maxTotalThreadsPerThreadgroup(self: *MTLComputePipelineState) usize {
        return msgSend(self, "maxTotalThreadsPerThreadgroup", .{}, usize);
    }

    pub fn threadExecutionWidth(self: *MTLComputePipelineState) usize {
        return msgSend(self, "threadExecutionWidth", .{}, usize);
    }
};

pub const NSString = opaque {
    pub fn initWithUTF8String(str: [:0]const u8) *NSString {
        const alloc = classMsgSend("NSString", "alloc", .{}, *NSString);
        return msgSend(alloc, "initWithUTF8String:", .{str.ptr}, *NSString);
    }

    pub fn UTF8String(self: *NSString) [*:0]const u8 {
        return msgSend(self, "UTF8String", .{}, [*:0]const u8);
    }
};

pub const NSError = opaque {
    pub fn localizedDescription(self: *NSError) *NSString {
        return msgSend(self, "localizedDescription", .{}, *NSString);
    }
};

pub const MTLResourceOptions = u64;
pub const MTLResourceStorageModeShared: MTLResourceOptions = 0 << 4;
pub const MTLResourceStorageModeManaged: MTLResourceOptions = 1 << 4;
pub const MTLResourceStorageModePrivate: MTLResourceOptions = 2 << 4;
pub const MTLResourceCPUCacheModeDefaultCache: MTLResourceOptions = 0 << 0;
pub const MTLResourceCPUCacheModeWriteCombined: MTLResourceOptions = 1 << 0;

pub const MTLSize = extern struct {
    width: usize,
    height: usize,
    depth: usize,

    pub fn init(w: usize, h: usize, d: usize) MTLSize {
        return .{ .width = w, .height = h, .depth = d };
    }

    pub fn init1D(w: usize) MTLSize {
        return init(w, 1, 1);
    }
};

pub fn createSystemDefaultDevice() ?*MTLDevice {
    const MTLCreateSystemDefaultDevice = @extern(*const fn () callconv(std.builtin.CallingConvention.c) ?*MTLDevice, .{
        .name = "MTLCreateSystemDefaultDevice",
    });
    return MTLCreateSystemDefaultDevice();
}
