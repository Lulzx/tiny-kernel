const std = @import("std");
const tk = @import("main.zig");

const Timer = struct {
    start: i128,

    fn now() Timer {
        return .{ .start = std.time.nanoTimestamp() };
    }

    fn elapsedMs(self: Timer) f64 {
        const end = std.time.nanoTimestamp();
        return @as(f64, @floatFromInt(end - self.start)) / 1_000_000.0;
    }
};

fn benchRMSNorm(device: *tk.Device, iterations: usize) !void {
    const hidden_dim: usize = 4096;
    const batch: usize = 32;

    var x = try device.createBuffer(f32, batch * hidden_dim);
    defer x.deinit();
    var weight = try device.createBuffer(f32, hidden_dim);
    defer weight.deinit();
    var out = try device.createBuffer(f32, batch * hidden_dim);
    defer out.deinit();

    for (x.slice()) |*v| v.* = 0.1;
    for (weight.slice()) |*v| v.* = 1.0;

    var rmsnorm = try tk.kernels.rmsnorm.RMSNorm.init(device);

    for (0..10) |_| {
        try rmsnorm.forward(&x, &weight, &out, @intCast(hidden_dim), 1e-6);
    }

    const timer = Timer.now();
    for (0..iterations) |_| {
        try rmsnorm.forward(&x, &weight, &out, @intCast(hidden_dim), 1e-6);
    }
    const elapsed = timer.elapsedMs();

    std.debug.print("RMSNorm [{d}x{d}]: {d:.3} ms/iter\n", .{ batch, hidden_dim, elapsed / @as(f64, @floatFromInt(iterations)) });
}

fn benchFused(device: *tk.Device, iterations: usize) !void {
    const hidden_dim: usize = 4096;
    const inter_dim: usize = 14336;
    const head_dim: usize = 128;
    const batch: usize = 1;
    const max_seq: usize = 8192;

    var x = try device.createBuffer(f32, batch * hidden_dim);
    defer x.deinit();
    var norm_weight = try device.createBuffer(f32, hidden_dim);
    defer norm_weight.deinit();
    var cos_cache = try device.createBuffer(f32, max_seq * head_dim / 2);
    defer cos_cache.deinit();
    var sin_cache = try device.createBuffer(f32, max_seq * head_dim / 2);
    defer sin_cache.deinit();
    var gate_weight = try device.createBuffer(f32, hidden_dim * inter_dim);
    defer gate_weight.deinit();
    var up_weight = try device.createBuffer(f32, hidden_dim * inter_dim);
    defer up_weight.deinit();
    var out = try device.createBuffer(f32, batch * inter_dim);
    defer out.deinit();

    for (x.slice()) |*v| v.* = 0.1;
    for (norm_weight.slice()) |*v| v.* = 1.0;
    for (cos_cache.slice()) |*v| v.* = 1.0;
    for (sin_cache.slice()) |*v| v.* = 0.0;
    for (gate_weight.slice()) |*v| v.* = 0.01;
    for (up_weight.slice()) |*v| v.* = 0.01;

    var fused = try tk.kernels.fused.FusedRMSNormRoPESwiGLU.init(device);

    for (0..10) |_| {
        try fused.forward(&x, &norm_weight, &cos_cache, &sin_cache, &gate_weight, &up_weight, &out, @intCast(hidden_dim), @intCast(inter_dim), @intCast(head_dim), 0, 1e-6);
    }

    const timer = Timer.now();
    for (0..iterations) |_| {
        try fused.forward(&x, &norm_weight, &cos_cache, &sin_cache, &gate_weight, &up_weight, &out, @intCast(hidden_dim), @intCast(inter_dim), @intCast(head_dim), 0, 1e-6);
    }
    const elapsed = timer.elapsedMs();

    std.debug.print("Fused RMSNorm+RoPE+SwiGLU [{d}x{d}->{d}]: {d:.3} ms/iter\n", .{ batch, hidden_dim, inter_dim, elapsed / @as(f64, @floatFromInt(iterations)) });
}

fn benchFlashAttention(device: *tk.Device, iterations: usize) !void {
    const seq_len: usize = 512;
    const head_dim: usize = 128;
    const batch_heads: usize = 32;

    var q = try device.createBuffer(f32, batch_heads * seq_len * head_dim);
    defer q.deinit();
    var k = try device.createBuffer(f32, batch_heads * seq_len * head_dim);
    defer k.deinit();
    var v = try device.createBuffer(f32, batch_heads * seq_len * head_dim);
    defer v.deinit();
    var o = try device.createBuffer(f32, batch_heads * seq_len * head_dim);
    defer o.deinit();

    for (q.slice()) |*val| val.* = 0.1;
    for (k.slice()) |*val| val.* = 0.1;
    for (v.slice()) |*val| val.* = 0.1;

    var attn = try tk.kernels.attention.FlashAttention.init(device);

    for (0..10) |_| {
        try attn.forward(&q, &k, &v, &o, @intCast(batch_heads), @intCast(seq_len), @intCast(head_dim));
    }

    const timer = Timer.now();
    for (0..iterations) |_| {
        try attn.forward(&q, &k, &v, &o, @intCast(batch_heads), @intCast(seq_len), @intCast(head_dim));
    }
    const elapsed = timer.elapsedMs();

    std.debug.print("Flash Attention [heads={d}, seq={d}, dim={d}]: {d:.3} ms/iter\n", .{ batch_heads, seq_len, head_dim, elapsed / @as(f64, @floatFromInt(iterations)) });
}

pub fn main() !void {
    const device = try tk.init();
    defer tk.deinit(device);

    std.debug.print("\ntiny-kernel benchmarks\n", .{});
    std.debug.print("Device: {s}\n", .{device.name()});
    std.debug.print("========================\n\n", .{});

    const iterations: usize = 100;

    try benchRMSNorm(device, iterations);
    try benchFused(device, iterations);
    try benchFlashAttention(device, iterations);

    std.debug.print("\nDone.\n", .{});
}
