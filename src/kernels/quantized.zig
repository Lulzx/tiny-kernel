const std = @import("std");
const metal = @import("../metal/metal.zig");
const shaders = @import("shaders.zig");
const objc = metal.objc;

pub const QMatMulInt4 = struct {
    device: *metal.Device,
    pipeline: *objc.MTLComputePipelineState,

    pub fn init(device: *metal.Device) !QMatMulInt4 {
        const pipeline = try device.compileKernel(shaders.qmatmul_int4, "qmatmul_int4");
        return .{ .device = device, .pipeline = pipeline };
    }

    pub fn forward(
        self: *QMatMulInt4,
        x: *metal.Buffer(f32),
        w_packed: *metal.Buffer(u8),
        scales: *metal.Buffer(f16),
        zeros: *metal.Buffer(u8),
        out: *metal.Buffer(f32),
        m: u32,
        k: u32,
        n: u32,
        group_size: u32,
    ) !void {
        var cmd = try self.device.createCommandBuffer();
        var encoder = try cmd.computeEncoder();

        encoder.setPipeline(self.pipeline);
        encoder.handle.setBuffer(x.handle, 0, 0);
        encoder.handle.setBuffer(w_packed.handle, 0, 1);
        encoder.handle.setBuffer(scales.handle, 0, 2);
        encoder.handle.setBuffer(zeros.handle, 0, 3);
        encoder.handle.setBuffer(out.handle, 0, 4);
        encoder.setBytes(u32, &m, 5);
        encoder.setBytes(u32, &k, 6);
        encoder.setBytes(u32, &n, 7);
        encoder.setBytes(u32, &group_size, 8);

        const n_blocks = (n + 255) / 256;
        encoder.dispatch(
            objc.MTLSize.init1D(n_blocks),
            objc.MTLSize.init1D(256),
        );

        encoder.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
    }
};
