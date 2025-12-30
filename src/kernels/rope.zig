const std = @import("std");
const metal = @import("../metal/metal.zig");
const shaders = @import("shaders.zig");
const objc = metal.objc;

pub const RoPE = struct {
    device: *metal.Device,
    pipeline: *objc.MTLComputePipelineState,

    pub fn init(device: *metal.Device) !RoPE {
        const pipeline = try device.compileKernel(shaders.rope, "rope");
        return .{ .device = device, .pipeline = pipeline };
    }

    pub fn forward(
        self: *RoPE,
        x: *metal.Buffer(f32),
        cos_cache: *metal.Buffer(f32),
        sin_cache: *metal.Buffer(f32),
        out: *metal.Buffer(f32),
        dim: u32,
        head_dim: u32,
        pos: u32,
    ) !void {
        var cmd = try self.device.createCommandBuffer();
        var encoder = try cmd.computeEncoder();

        encoder.setPipeline(self.pipeline);
        encoder.handle.setBuffer(x.handle, 0, 0);
        encoder.handle.setBuffer(cos_cache.handle, 0, 1);
        encoder.handle.setBuffer(sin_cache.handle, 0, 2);
        encoder.handle.setBuffer(out.handle, 0, 3);
        encoder.setBytes(u32, &dim, 4);
        encoder.setBytes(u32, &head_dim, 5);
        encoder.setBytes(u32, &pos, 6);

        const batch = x.count / dim;
        encoder.dispatch(
            objc.MTLSize.init(batch, 1, 1),
            objc.MTLSize.init(256, 1, 1),
        );

        encoder.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
    }
};
