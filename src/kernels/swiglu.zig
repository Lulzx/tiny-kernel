const std = @import("std");
const metal = @import("../metal/metal.zig");
const shaders = @import("shaders.zig");
const objc = metal.objc;

pub const SwiGLU = struct {
    device: *metal.Device,
    pipeline: *objc.MTLComputePipelineState,

    pub fn init(device: *metal.Device) !SwiGLU {
        const pipeline = try device.compileKernel(shaders.swiglu, "swiglu");
        return .{ .device = device, .pipeline = pipeline };
    }

    pub fn forward(
        self: *SwiGLU,
        x: *metal.Buffer(f32),
        gate_weight: *metal.Buffer(f32),
        up_weight: *metal.Buffer(f32),
        out: *metal.Buffer(f32),
        in_dim: u32,
        out_dim: u32,
    ) !void {
        var cmd = try self.device.createCommandBuffer();
        var encoder = try cmd.computeEncoder();

        encoder.setPipeline(self.pipeline);
        encoder.handle.setBuffer(x.handle, 0, 0);
        encoder.handle.setBuffer(gate_weight.handle, 0, 1);
        encoder.handle.setBuffer(up_weight.handle, 0, 2);
        encoder.handle.setBuffer(out.handle, 0, 3);
        encoder.setBytes(u32, &in_dim, 4);
        encoder.setBytes(u32, &out_dim, 5);

        const batch = x.count / in_dim;
        encoder.dispatch(
            objc.MTLSize.init(batch, 1, 1),
            objc.MTLSize.init(256, 1, 1),
        );

        encoder.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
    }
};
