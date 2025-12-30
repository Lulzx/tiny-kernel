const std = @import("std");
const metal = @import("../metal/metal.zig");
const shaders = @import("shaders.zig");
const objc = metal.objc;

pub const RMSNorm = struct {
    device: *metal.Device,
    pipeline: *objc.MTLComputePipelineState,

    pub fn init(device: *metal.Device) !RMSNorm {
        const pipeline = try device.compileKernel(shaders.rmsnorm, "rmsnorm");
        return .{ .device = device, .pipeline = pipeline };
    }

    pub fn forward(
        self: *RMSNorm,
        x: *metal.Buffer(f32),
        weight: *metal.Buffer(f32),
        out: *metal.Buffer(f32),
        dim: u32,
        eps: f32,
    ) !void {
        var cmd = try self.device.createCommandBuffer();
        var encoder = try cmd.computeEncoder();

        encoder.setPipeline(self.pipeline);
        encoder.handle.setBuffer(x.handle, 0, 0);
        encoder.handle.setBuffer(weight.handle, 0, 1);
        encoder.handle.setBuffer(out.handle, 0, 2);
        encoder.setBytes(u32, &dim, 3);
        encoder.setBytes(f32, &eps, 4);

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
