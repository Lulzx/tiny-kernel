const std = @import("std");
const metal = @import("../metal/metal.zig");
const shaders = @import("shaders.zig");
const objc = metal.objc;

pub const FusedRMSNormRoPESwiGLU = struct {
    device: *metal.Device,
    pipeline: *objc.MTLComputePipelineState,

    pub fn init(device: *metal.Device) !FusedRMSNormRoPESwiGLU {
        const pipeline = try device.compileKernel(shaders.fused_rmsnorm_rope_swiglu, "fused_rmsnorm_rope_swiglu");
        return .{ .device = device, .pipeline = pipeline };
    }

    pub fn forward(
        self: *FusedRMSNormRoPESwiGLU,
        x: *metal.Buffer(f32),
        norm_weight: *metal.Buffer(f32),
        cos_cache: *metal.Buffer(f32),
        sin_cache: *metal.Buffer(f32),
        gate_weight: *metal.Buffer(f32),
        up_weight: *metal.Buffer(f32),
        out: *metal.Buffer(f32),
        hidden_dim: u32,
        inter_dim: u32,
        head_dim: u32,
        pos: u32,
        eps: f32,
    ) !void {
        var cmd = try self.device.createCommandBuffer();
        var encoder = try cmd.computeEncoder();

        encoder.setPipeline(self.pipeline);
        encoder.handle.setBuffer(x.handle, 0, 0);
        encoder.handle.setBuffer(norm_weight.handle, 0, 1);
        encoder.handle.setBuffer(cos_cache.handle, 0, 2);
        encoder.handle.setBuffer(sin_cache.handle, 0, 3);
        encoder.handle.setBuffer(gate_weight.handle, 0, 4);
        encoder.handle.setBuffer(up_weight.handle, 0, 5);
        encoder.handle.setBuffer(out.handle, 0, 6);
        encoder.setBytes(u32, &hidden_dim, 7);
        encoder.setBytes(u32, &inter_dim, 8);
        encoder.setBytes(u32, &head_dim, 9);
        encoder.setBytes(u32, &pos, 10);
        encoder.setBytes(f32, &eps, 11);

        const batch = x.count / hidden_dim;
        encoder.dispatch(
            objc.MTLSize.init(batch, 1, 1),
            objc.MTLSize.init(256, 1, 1),
        );

        encoder.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
    }
};
