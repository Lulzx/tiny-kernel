const std = @import("std");
const metal = @import("../metal/metal.zig");
const shaders = @import("shaders.zig");
const objc = metal.objc;

pub const FlashAttention = struct {
    device: *metal.Device,
    pipeline: *objc.MTLComputePipelineState,

    pub fn init(device: *metal.Device) !FlashAttention {
        const pipeline = try device.compileKernel(shaders.flash_attention_fwd, "flash_attention_fwd");
        return .{ .device = device, .pipeline = pipeline };
    }

    pub fn forward(
        self: *FlashAttention,
        q: *metal.Buffer(f32),
        k: *metal.Buffer(f32),
        v: *metal.Buffer(f32),
        o: *metal.Buffer(f32),
        batch_heads: u32,
        seq_len: u32,
        head_dim: u32,
    ) !void {
        var cmd = try self.device.createCommandBuffer();
        var encoder = try cmd.computeEncoder();

        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

        encoder.setPipeline(self.pipeline);
        encoder.handle.setBuffer(q.handle, 0, 0);
        encoder.handle.setBuffer(k.handle, 0, 1);
        encoder.handle.setBuffer(v.handle, 0, 2);
        encoder.handle.setBuffer(o.handle, 0, 3);
        encoder.setBytes(u32, &seq_len, 4);
        encoder.setBytes(u32, &head_dim, 5);
        encoder.setBytes(f32, &scale, 6);

        const q_blocks = (seq_len + 15) / 16;
        encoder.dispatch(
            objc.MTLSize.init1D(batch_heads * q_blocks),
            objc.MTLSize.init1D(256),
        );

        encoder.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
    }
};

pub const PagedAttention = struct {
    device: *metal.Device,
    pipeline: *objc.MTLComputePipelineState,

    pub fn init(device: *metal.Device) !PagedAttention {
        const pipeline = try device.compileKernel(shaders.paged_attention, "paged_attention");
        return .{ .device = device, .pipeline = pipeline };
    }

    pub fn forward(
        self: *PagedAttention,
        q: *metal.Buffer(f32),
        k_cache: *metal.Buffer(f32),
        v_cache: *metal.Buffer(f32),
        block_tables: *metal.Buffer(i32),
        context_lens: *metal.Buffer(i32),
        o: *metal.Buffer(f32),
        batch_size: u32,
        num_heads: u32,
        head_dim: u32,
        max_blocks: u32,
    ) !void {
        var cmd = try self.device.createCommandBuffer();
        var encoder = try cmd.computeEncoder();

        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

        encoder.setPipeline(self.pipeline);
        encoder.handle.setBuffer(q.handle, 0, 0);
        encoder.handle.setBuffer(k_cache.handle, 0, 1);
        encoder.handle.setBuffer(v_cache.handle, 0, 2);
        encoder.handle.setBuffer(block_tables.handle, 0, 3);
        encoder.handle.setBuffer(context_lens.handle, 0, 4);
        encoder.handle.setBuffer(o.handle, 0, 5);
        encoder.setBytes(u32, &num_heads, 6);
        encoder.setBytes(u32, &head_dim, 7);
        encoder.setBytes(u32, &max_blocks, 8);
        encoder.setBytes(f32, &scale, 9);

        encoder.dispatch(
            objc.MTLSize.init(num_heads, batch_size, 1),
            objc.MTLSize.init(256, 1, 1),
        );

        encoder.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
    }
};

pub const GQA = struct {
    device: *metal.Device,
    pipeline: *objc.MTLComputePipelineState,

    pub fn init(device: *metal.Device) !GQA {
        const pipeline = try device.compileKernel(shaders.gqa_attention, "gqa_attention");
        return .{ .device = device, .pipeline = pipeline };
    }

    pub fn forward(
        self: *GQA,
        q: *metal.Buffer(f32),
        k: *metal.Buffer(f32),
        v: *metal.Buffer(f32),
        o: *metal.Buffer(f32),
        batch_size: u32,
        num_heads: u32,
        num_kv_heads: u32,
        seq_len: u32,
        head_dim: u32,
    ) !void {
        var cmd = try self.device.createCommandBuffer();
        var encoder = try cmd.computeEncoder();

        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

        encoder.setPipeline(self.pipeline);
        encoder.handle.setBuffer(q.handle, 0, 0);
        encoder.handle.setBuffer(k.handle, 0, 1);
        encoder.handle.setBuffer(v.handle, 0, 2);
        encoder.handle.setBuffer(o.handle, 0, 3);
        encoder.setBytes(u32, &batch_size, 4);
        encoder.setBytes(u32, &num_heads, 5);
        encoder.setBytes(u32, &num_kv_heads, 6);
        encoder.setBytes(u32, &seq_len, 7);
        encoder.setBytes(u32, &head_dim, 8);
        encoder.setBytes(f32, &scale, 9);

        const q_blocks = (seq_len + 15) / 16;
        encoder.dispatch(
            objc.MTLSize.init(q_blocks, num_heads, batch_size),
            objc.MTLSize.init(256, 1, 1),
        );

        encoder.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
    }
};
