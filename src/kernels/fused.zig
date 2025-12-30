const std = @import("std");
const metal = @import("../metal/metal.zig");
const shaders = @import("shaders.zig");
const objc = metal.objc;

pub const FusedRMSNormRoPE = struct {
    device: *metal.Device,
    pipeline: *objc.MTLComputePipelineState,

    pub fn init(device: *metal.Device) !FusedRMSNormRoPE {
        const pipeline = try device.compileKernel(shaders.fused_rmsnorm_rope, "fused_rmsnorm_rope");
        return .{ .device = device, .pipeline = pipeline };
    }

    pub fn forward(
        self: *FusedRMSNormRoPE,
        x: *metal.Buffer(f32),
        weight: *metal.Buffer(f32),
        cos_cache: *metal.Buffer(f32),
        sin_cache: *metal.Buffer(f32),
        out: *metal.Buffer(f32),
        dim: u32,
        head_dim: u32,
        pos: u32,
        eps: f32,
    ) !void {
        var cmd = try self.device.createCommandBuffer();
        var encoder = try cmd.computeEncoder();

        encoder.setPipeline(self.pipeline);
        encoder.handle.setBuffer(x.handle, 0, 0);
        encoder.handle.setBuffer(weight.handle, 0, 1);
        encoder.handle.setBuffer(cos_cache.handle, 0, 2);
        encoder.handle.setBuffer(sin_cache.handle, 0, 3);
        encoder.handle.setBuffer(out.handle, 0, 4);
        encoder.setBytes(u32, &dim, 5);
        encoder.setBytes(u32, &head_dim, 6);
        encoder.setBytes(u32, &pos, 7);
        encoder.setBytes(f32, &eps, 8);

        const batch = x.count / dim;
        encoder.dispatch(
            objc.MTLSize.init1D(batch),
            objc.MTLSize.init1D(256),
        );

        encoder.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
    }
};

pub const Matvec = struct {
    device: *metal.Device,
    pipeline: *objc.MTLComputePipelineState,

    pub fn init(device: *metal.Device) !Matvec {
        const pipeline = try device.compileKernel(shaders.matvec, "matvec");
        return .{ .device = device, .pipeline = pipeline };
    }

    pub fn forward(
        self: *Matvec,
        x: *metal.Buffer(f32),
        W: *metal.Buffer(f32),
        out: *metal.Buffer(f32),
        K: u32,
        N: u32,
    ) !void {
        var cmd = try self.device.createCommandBuffer();
        var encoder = try cmd.computeEncoder();

        encoder.setPipeline(self.pipeline);
        encoder.handle.setBuffer(x.handle, 0, 0);
        encoder.handle.setBuffer(W.handle, 0, 1);
        encoder.handle.setBuffer(out.handle, 0, 2);
        encoder.setBytes(u32, &K, 3);
        encoder.setBytes(u32, &N, 4);

        const outputs_per_group = 256 * 4;
        const num_groups = (N + outputs_per_group - 1) / outputs_per_group;
        encoder.dispatch(
            objc.MTLSize.init1D(num_groups),
            objc.MTLSize.init1D(256),
        );

        encoder.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
    }
};

pub const MatvecDual = struct {
    device: *metal.Device,
    pipeline: *objc.MTLComputePipelineState,

    pub fn init(device: *metal.Device) !MatvecDual {
        const pipeline = try device.compileKernel(shaders.matvec_dual, "matvec_dual");
        return .{ .device = device, .pipeline = pipeline };
    }

    pub fn forward(
        self: *MatvecDual,
        x: *metal.Buffer(f32),
        W1: *metal.Buffer(f32),
        W2: *metal.Buffer(f32),
        out1: *metal.Buffer(f32),
        out2: *metal.Buffer(f32),
        K: u32,
        N: u32,
    ) !void {
        var cmd = try self.device.createCommandBuffer();
        var encoder = try cmd.computeEncoder();

        encoder.setPipeline(self.pipeline);
        encoder.handle.setBuffer(x.handle, 0, 0);
        encoder.handle.setBuffer(W1.handle, 0, 1);
        encoder.handle.setBuffer(W2.handle, 0, 2);
        encoder.handle.setBuffer(out1.handle, 0, 3);
        encoder.handle.setBuffer(out2.handle, 0, 4);
        encoder.setBytes(u32, &K, 5);
        encoder.setBytes(u32, &N, 6);

        const num_groups = (N + 7) / 8;
        encoder.dispatch(
            objc.MTLSize.init1D(num_groups),
            objc.MTLSize.init1D(256),
        );

        encoder.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
    }
};

pub const TiledMatmul = struct {
    device: *metal.Device,
    pipeline: *objc.MTLComputePipelineState,

    pub fn init(device: *metal.Device) !TiledMatmul {
        const pipeline = try device.compileKernel(shaders.simdgroup_matmul, "simdgroup_matmul");
        return .{ .device = device, .pipeline = pipeline };
    }

    pub fn forward(
        self: *TiledMatmul,
        a: *metal.Buffer(f32),
        b: *metal.Buffer(f32),
        c: *metal.Buffer(f32),
        M: u32,
        N: u32,
        K: u32,
    ) !void {
        var cmd = try self.device.createCommandBuffer();
        var encoder = try cmd.computeEncoder();

        encoder.setPipeline(self.pipeline);
        encoder.handle.setBuffer(a.handle, 0, 0);
        encoder.handle.setBuffer(b.handle, 0, 1);
        encoder.handle.setBuffer(c.handle, 0, 2);
        encoder.setBytes(u32, &M, 3);
        encoder.setBytes(u32, &N, 4);
        encoder.setBytes(u32, &K, 5);

        const tiles_m = (M + 31) / 32;
        const tiles_n = (N + 31) / 32;
        encoder.dispatch(
            objc.MTLSize.init(tiles_n, tiles_m, 1),
            objc.MTLSize.init(32, 8, 1),
        );

        encoder.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
    }
};

pub const FusedSiluMul = struct {
    device: *metal.Device,
    pipeline: *objc.MTLComputePipelineState,

    pub fn init(device: *metal.Device) !FusedSiluMul {
        const pipeline = try device.compileKernel(shaders.fused_silu_mul, "fused_silu_mul");
        return .{ .device = device, .pipeline = pipeline };
    }

    pub fn forward(
        self: *FusedSiluMul,
        gate: *metal.Buffer(f32),
        up: *metal.Buffer(f32),
        out: *metal.Buffer(f32),
    ) !void {
        var cmd = try self.device.createCommandBuffer();
        var encoder = try cmd.computeEncoder();

        const n: u32 = @intCast(gate.count);

        encoder.setPipeline(self.pipeline);
        encoder.handle.setBuffer(gate.handle, 0, 0);
        encoder.handle.setBuffer(up.handle, 0, 1);
        encoder.handle.setBuffer(out.handle, 0, 2);
        encoder.setBytes(u32, &n, 3);

        encoder.dispatch(
            objc.MTLSize.init1D((n + 255) / 256 * 256),
            objc.MTLSize.init1D(256),
        );

        encoder.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
    }
};

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
