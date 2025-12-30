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

fn benchFusedSlow(device: *tk.Device, iterations: usize) !void {
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

    std.debug.print("Naive fused (SLOW) [{d}x{d}->{d}]: {d:.3} ms/iter\n", .{ batch, hidden_dim, inter_dim, elapsed / @as(f64, @floatFromInt(iterations)) });
}

fn benchFusedFast(device: *tk.Device, iterations: usize) !void {
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
    var normed = try device.createBuffer(f32, batch * hidden_dim);
    defer normed.deinit();
    var gate_weight = try device.createBuffer(f32, hidden_dim * inter_dim);
    defer gate_weight.deinit();
    var up_weight = try device.createBuffer(f32, hidden_dim * inter_dim);
    defer up_weight.deinit();
    var gate_out = try device.createBuffer(f32, batch * inter_dim);
    defer gate_out.deinit();
    var up_out = try device.createBuffer(f32, batch * inter_dim);
    defer up_out.deinit();
    var out = try device.createBuffer(f32, batch * inter_dim);
    defer out.deinit();

    for (x.slice()) |*v| v.* = 0.1;
    for (norm_weight.slice()) |*v| v.* = 1.0;
    for (cos_cache.slice()) |*v| v.* = 1.0;
    for (sin_cache.slice()) |*v| v.* = 0.0;
    for (gate_weight.slice()) |*v| v.* = 0.01;
    for (up_weight.slice()) |*v| v.* = 0.01;

    var fused_norm_rope = try tk.kernels.fused.FusedRMSNormRoPE.init(device);
    var matvec = try tk.kernels.fused.Matvec.init(device);
    var silu_mul = try tk.kernels.fused.FusedSiluMul.init(device);

    for (0..10) |_| {
        try fused_norm_rope.forward(&x, &norm_weight, &cos_cache, &sin_cache, &normed, @intCast(hidden_dim), @intCast(head_dim), 0, 1e-6);
        try matvec.forward(&normed, &gate_weight, &gate_out, @intCast(hidden_dim), @intCast(inter_dim));
        try matvec.forward(&normed, &up_weight, &up_out, @intCast(hidden_dim), @intCast(inter_dim));
        try silu_mul.forward(&gate_out, &up_out, &out);
    }

    const timer = Timer.now();
    for (0..iterations) |_| {
        try fused_norm_rope.forward(&x, &norm_weight, &cos_cache, &sin_cache, &normed, @intCast(hidden_dim), @intCast(head_dim), 0, 1e-6);
        try matvec.forward(&normed, &gate_weight, &gate_out, @intCast(hidden_dim), @intCast(inter_dim));
        try matvec.forward(&normed, &up_weight, &up_out, @intCast(hidden_dim), @intCast(inter_dim));
        try silu_mul.forward(&gate_out, &up_out, &out);
    }
    const elapsed = timer.elapsedMs();

    std.debug.print("Split fusion (FAST) [{d}x{d}->{d}]: {d:.3} ms/iter\n", .{ batch, hidden_dim, inter_dim, elapsed / @as(f64, @floatFromInt(iterations)) });
}

fn benchMatvec(device: *tk.Device, iterations: usize) !void {
    const K: usize = 4096;
    const N: usize = 14336;

    var x = try device.createBuffer(f32, K);
    defer x.deinit();
    var W = try device.createBuffer(f32, K * N);
    defer W.deinit();
    var out = try device.createBuffer(f32, N);
    defer out.deinit();

    for (x.slice()) |*v| v.* = 0.1;
    for (W.slice()) |*v| v.* = 0.01;

    var matvec = try tk.kernels.fused.Matvec.init(device);

    for (0..10) |_| {
        try matvec.forward(&x, &W, &out, @intCast(K), @intCast(N));
    }

    const timer = Timer.now();
    for (0..iterations) |_| {
        try matvec.forward(&x, &W, &out, @intCast(K), @intCast(N));
    }
    const elapsed = timer.elapsedMs();

    const flops = 2.0 * @as(f64, @floatFromInt(K * N));
    const time_per_iter = elapsed / @as(f64, @floatFromInt(iterations));
    const gflops = flops / (time_per_iter * 1e6);

    std.debug.print("Matvec [{d}x{d}]: {d:.3} ms/iter ({d:.1} GFLOPS)\n", .{ K, N, time_per_iter, gflops });
}

fn benchMatvecDual(device: *tk.Device, iterations: usize) !void {
    const K: usize = 4096;
    const N: usize = 14336;

    var x = try device.createBuffer(f32, K);
    defer x.deinit();
    var W1 = try device.createBuffer(f32, K * N);
    defer W1.deinit();
    var W2 = try device.createBuffer(f32, K * N);
    defer W2.deinit();
    var out1 = try device.createBuffer(f32, N);
    defer out1.deinit();
    var out2 = try device.createBuffer(f32, N);
    defer out2.deinit();

    for (x.slice()) |*v| v.* = 0.1;
    for (W1.slice()) |*v| v.* = 0.01;
    for (W2.slice()) |*v| v.* = 0.01;

    var matvec_dual = try tk.kernels.fused.MatvecDual.init(device);

    for (0..10) |_| {
        try matvec_dual.forward(&x, &W1, &W2, &out1, &out2, @intCast(K), @intCast(N));
    }

    const timer = Timer.now();
    for (0..iterations) |_| {
        try matvec_dual.forward(&x, &W1, &W2, &out1, &out2, @intCast(K), @intCast(N));
    }
    const elapsed = timer.elapsedMs();

    const flops = 4.0 * @as(f64, @floatFromInt(K * N));
    const time_per_iter = elapsed / @as(f64, @floatFromInt(iterations));
    const gflops = flops / (time_per_iter * 1e6);

    std.debug.print("Matvec Dual [{d}x{d}]: {d:.3} ms/iter ({d:.1} GFLOPS)\n", .{ K, N, time_per_iter, gflops });
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
    try benchMatvec(device, iterations);
    try benchMatvecDual(device, iterations);
    try benchFusedSlow(device, iterations);
    try benchFusedFast(device, iterations);
    try benchFlashAttention(device, iterations);

    std.debug.print("\nDone.\n", .{});
}
