const std = @import("std");
const gguf = @import("gguf.zig");
const metal = @import("../metal/metal.zig");
const kernels = @import("../kernels/kernels.zig");

pub const LlamaConfig = struct {
    dim: u32,
    hidden_dim: u32,
    n_layers: u32,
    n_heads: u32,
    n_kv_heads: u32,
    vocab_size: u32,
    max_seq_len: u32,
    head_dim: u32,
    kv_dim: u32,
    rope_theta: f32,
    rms_norm_eps: f32,

    pub fn fromGGUF(g: *gguf.GGUFFile) !LlamaConfig {
        const dim = g.getMetaU32("llama.embedding_length") orelse return error.MissingDim;
        const n_heads = g.getMetaU32("llama.attention.head_count") orelse return error.MissingHeads;
        const n_kv_heads = g.getMetaU32("llama.attention.head_count_kv") orelse n_heads;

        var vocab_size: u32 = 32000;
        if (g.getMetaArray("tokenizer.ggml.tokens")) |arr| {
            vocab_size = @intCast(arr.len);
        }

        const head_dim = dim / n_heads;
        const kv_dim = head_dim * n_kv_heads;

        return .{
            .dim = dim,
            .hidden_dim = g.getMetaU32("llama.feed_forward_length") orelse dim * 4,
            .n_layers = g.getMetaU32("llama.block_count") orelse return error.MissingLayers,
            .n_heads = n_heads,
            .n_kv_heads = n_kv_heads,
            .vocab_size = vocab_size,
            .max_seq_len = g.getMetaU32("llama.context_length") orelse 2048,
            .head_dim = head_dim,
            .kv_dim = kv_dim,
            .rope_theta = g.getMetaF32("llama.rope.freq_base") orelse 10000.0,
            .rms_norm_eps = g.getMetaF32("llama.attention.layer_norm_rms_epsilon") orelse 1e-5,
        };
    }
};

const Q4Block = extern struct {
    d: f16,
    qs: [16]u8,
};

const Q8Block = extern struct {
    d: f16,
    qs: [32]i8,
};

pub const Llama = struct {
    allocator: std.mem.Allocator,
    config: LlamaConfig,
    device: *metal.Device,
    gguf_file: *gguf.GGUFFile,

    token_embed: metal.Buffer(f32),
    output_norm: metal.Buffer(f32),
    output_weight: metal.Buffer(f32),

    layers: []Layer,

    x: metal.Buffer(f32),
    xb: metal.Buffer(f32),
    q: metal.Buffer(f32),
    k: metal.Buffer(f32),
    v: metal.Buffer(f32),
    attn_out: metal.Buffer(f32),
    ffn_gate: metal.Buffer(f32),
    ffn_up: metal.Buffer(f32),
    ffn_out: metal.Buffer(f32),
    logits: metal.Buffer(f32),

    k_cache: metal.Buffer(f32),
    v_cache: metal.Buffer(f32),
    cos_cache: metal.Buffer(f32),
    sin_cache: metal.Buffer(f32),

    token_buf: metal.Buffer(u32),
    sample_buf: metal.Buffer(u32),

    rmsnorm: kernels.rmsnorm.RMSNorm,
    rope: kernels.rope.RoPE,
    matvec: kernels.fused.Matvec,
    silu_mul: kernels.fused.FusedSiluMul,

    const Layer = struct {
        attn_norm: metal.Buffer(f32),
        wq: metal.Buffer(f32),
        wk: metal.Buffer(f32),
        wv: metal.Buffer(f32),
        wo: metal.Buffer(f32),
        ffn_norm: metal.Buffer(f32),
        w1: metal.Buffer(f32),
        w2: metal.Buffer(f32),
        w3: metal.Buffer(f32),
    };

    pub fn init(allocator: std.mem.Allocator, device: *metal.Device, path: []const u8) !*Llama {
        var g = try gguf.GGUFFile.open(allocator, path);
        const config = try LlamaConfig.fromGGUF(g);

        std.debug.print("Model config:\n", .{});
        std.debug.print("  dim: {}\n", .{config.dim});
        std.debug.print("  hidden_dim: {}\n", .{config.hidden_dim});
        std.debug.print("  n_layers: {}\n", .{config.n_layers});
        std.debug.print("  n_heads: {}\n", .{config.n_heads});
        std.debug.print("  n_kv_heads: {}\n", .{config.n_kv_heads});
        std.debug.print("  vocab_size: {}\n", .{config.vocab_size});
        std.debug.print("  max_seq_len: {}\n", .{config.max_seq_len});

        var self = try allocator.create(Llama);
        self.allocator = allocator;
        self.config = config;
        self.device = device;
        self.gguf_file = g;

        self.token_embed = try loadF32Tensor(device, g, "token_embd.weight", config.vocab_size * config.dim);
        self.output_norm = try loadF32Tensor(device, g, "output_norm.weight", config.dim);

        if (g.getTensor("output.weight")) |_| {
            self.output_weight = try loadF32Tensor(device, g, "output.weight", config.vocab_size * config.dim);
        } else {
            self.output_weight = self.token_embed;
        }

        self.layers = try allocator.alloc(Layer, config.n_layers);
        for (0..config.n_layers) |i| {
            self.layers[i] = try loadLayer(device, g, config, @intCast(i));
        }

        self.x = try device.createBuffer(f32, config.dim);
        self.xb = try device.createBuffer(f32, config.dim);
        self.q = try device.createBuffer(f32, config.dim);
        self.k = try device.createBuffer(f32, config.kv_dim);
        self.v = try device.createBuffer(f32, config.kv_dim);
        self.attn_out = try device.createBuffer(f32, config.dim);
        self.ffn_gate = try device.createBuffer(f32, config.hidden_dim);
        self.ffn_up = try device.createBuffer(f32, config.hidden_dim);
        self.ffn_out = try device.createBuffer(f32, config.dim);
        self.logits = try device.createBuffer(f32, config.vocab_size);

        const cache_size = config.n_layers * config.n_kv_heads * config.max_seq_len * config.head_dim;
        self.k_cache = try device.createBuffer(f32, cache_size);
        self.v_cache = try device.createBuffer(f32, cache_size);

        for (self.k_cache.slice()) |*v| v.* = 0;
        for (self.v_cache.slice()) |*v| v.* = 0;

        self.cos_cache = try device.createBuffer(f32, config.max_seq_len * config.head_dim / 2);
        self.sin_cache = try device.createBuffer(f32, config.max_seq_len * config.head_dim / 2);
        precomputeRoPE(self.cos_cache.slice(), self.sin_cache.slice(), config.head_dim, config.max_seq_len, config.rope_theta);

        self.token_buf = try device.createBuffer(u32, 1);
        self.sample_buf = try device.createBuffer(u32, 1);

        self.rmsnorm = try kernels.rmsnorm.RMSNorm.init(device);
        self.rope = try kernels.rope.RoPE.init(device);
        self.matvec = try kernels.fused.Matvec.init(device);
        self.silu_mul = try kernels.fused.FusedSiluMul.init(device);

        return self;
    }

    pub fn deinit(self: *Llama) void {
        self.gguf_file.close();
        self.allocator.free(self.layers);
        self.allocator.destroy(self);
    }

    pub fn forward(self: *Llama, token: u32, pos: u32) !void {
        const c = self.config;

        const embed_slice = self.token_embed.slice();
        const x_slice = self.x.slice();
        const start = token * c.dim;
        @memcpy(x_slice, embed_slice[start .. start + c.dim]);

        for (0..c.n_layers) |layer_idx| {
            const layer = &self.layers[layer_idx];

            try self.rmsnorm.forward(&self.x, &layer.attn_norm, &self.xb, c.dim, c.rms_norm_eps);

            try self.matvec.forward(&self.xb, &layer.wq, &self.q, c.dim, c.dim);
            try self.matvec.forward(&self.xb, &layer.wk, &self.k, c.dim, c.kv_dim);
            try self.matvec.forward(&self.xb, &layer.wv, &self.v, c.dim, c.kv_dim);

            try self.rope.forward(&self.q, &self.cos_cache, &self.sin_cache, &self.q, c.dim, c.head_dim, pos);
            try self.rope.forward(&self.k, &self.cos_cache, &self.sin_cache, &self.k, c.kv_dim, c.head_dim, pos);

            const layer_cache_offset = layer_idx * c.n_kv_heads * c.max_seq_len * c.head_dim;
            const pos_offset = pos * c.head_dim;

            const k_cache_slice = self.k_cache.slice();
            const v_cache_slice = self.v_cache.slice();
            const k_slice = self.k.slice();
            const v_slice = self.v.slice();

            for (0..c.n_kv_heads) |h| {
                const head_offset = layer_cache_offset + h * c.max_seq_len * c.head_dim + pos_offset;
                @memcpy(k_cache_slice[head_offset .. head_offset + c.head_dim], k_slice[h * c.head_dim .. (h + 1) * c.head_dim]);
                @memcpy(v_cache_slice[head_offset .. head_offset + c.head_dim], v_slice[h * c.head_dim .. (h + 1) * c.head_dim]);
            }

            self.simpleAttention(layer_idx, pos);

            try self.matvec.forward(&self.attn_out, &layer.wo, &self.xb, c.dim, c.dim);

            for (x_slice, self.xb.slice()) |*xv, xbv| {
                xv.* += xbv;
            }

            try self.rmsnorm.forward(&self.x, &layer.ffn_norm, &self.xb, c.dim, c.rms_norm_eps);

            try self.matvec.forward(&self.xb, &layer.w1, &self.ffn_gate, c.dim, c.hidden_dim);
            try self.matvec.forward(&self.xb, &layer.w3, &self.ffn_up, c.dim, c.hidden_dim);

            try self.silu_mul.forward(&self.ffn_gate, &self.ffn_up, &self.ffn_out);

            try self.matvec.forward(&self.ffn_out, &layer.w2, &self.xb, c.hidden_dim, c.dim);

            for (x_slice, self.xb.slice()) |*xv, xbv| {
                xv.* += xbv;
            }
        }

        try self.rmsnorm.forward(&self.x, &self.output_norm, &self.xb, c.dim, c.rms_norm_eps);
        try self.matvec.forward(&self.xb, &self.output_weight, &self.logits, c.dim, c.vocab_size);
    }

    fn simpleAttention(self: *Llama, layer_idx: usize, pos: u32) void {
        const c = self.config;
        const kv_len = pos + 1;
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(c.head_dim)));

        const layer_cache_offset = layer_idx * c.n_kv_heads * c.max_seq_len * c.head_dim;
        const k_cache = self.k_cache.slice();
        const v_cache = self.v_cache.slice();
        const q_slice = self.q.slice();
        var attn_out = self.attn_out.slice();

        for (attn_out) |*v| v.* = 0;

        for (0..c.n_heads) |h| {
            const kv_head = h / (c.n_heads / c.n_kv_heads);
            const q_head = q_slice[h * c.head_dim .. (h + 1) * c.head_dim];
            const kv_cache_base = layer_cache_offset + kv_head * c.max_seq_len * c.head_dim;

            var scores: [2048]f32 = undefined;
            var max_score: f32 = -std.math.inf(f32);

            for (0..kv_len) |t| {
                var score: f32 = 0;
                const k_head = k_cache[kv_cache_base + t * c.head_dim .. kv_cache_base + (t + 1) * c.head_dim];
                for (0..c.head_dim) |d| {
                    score += q_head[d] * k_head[d];
                }
                score *= scale;
                scores[t] = score;
                max_score = @max(max_score, score);
            }

            var sum: f32 = 0;
            for (0..kv_len) |t| {
                scores[t] = @exp(scores[t] - max_score);
                sum += scores[t];
            }
            for (0..kv_len) |t| {
                scores[t] /= sum;
            }

            for (0..c.head_dim) |d| {
                var acc: f32 = 0;
                for (0..kv_len) |t| {
                    const v_head = v_cache[kv_cache_base + t * c.head_dim .. kv_cache_base + (t + 1) * c.head_dim];
                    acc += scores[t] * v_head[d];
                }
                attn_out[h * c.head_dim + d] = acc;
            }
        }
    }

    pub fn sample(self: *Llama, temperature: f32) u32 {
        const logits = self.logits.slice();

        if (temperature == 0.0) {
            var max_idx: u32 = 0;
            var max_val = logits[0];
            for (logits[1..], 1..) |v, i| {
                if (v > max_val) {
                    max_val = v;
                    max_idx = @intCast(i);
                }
            }
            return max_idx;
        }

        var max_val: f32 = -std.math.inf(f32);
        for (logits) |v| max_val = @max(max_val, v);

        var sum: f32 = 0;
        for (logits) |*v| {
            v.* = @exp((v.* - max_val) / temperature);
            sum += v.*;
        }
        for (logits) |*v| v.* /= sum;

        var rng = std.Random.DefaultPrng.init(@intCast(std.time.nanoTimestamp()));
        const r = rng.random().float(f32);
        var cumsum: f32 = 0;
        for (logits, 0..) |p, i| {
            cumsum += p;
            if (r < cumsum) return @intCast(i);
        }
        return @intCast(logits.len - 1);
    }

    fn loadF32Tensor(device: *metal.Device, g: *gguf.GGUFFile, name: []const u8, expected_size: usize) !metal.Buffer(f32) {
        const info = g.getTensor(name) orelse return error.TensorNotFound;
        const data = g.getTensorData(info);
        var buf = try device.createBuffer(f32, expected_size);

        switch (info.dtype) {
            .F32 => {
                const f32_data: [*]const f32 = @ptrCast(@alignCast(data.ptr));
                @memcpy(buf.slice(), f32_data[0..expected_size]);
            },
            .F16 => {
                const f16_data: [*]const f16 = @ptrCast(@alignCast(data.ptr));
                for (buf.slice(), 0..) |*v, i| {
                    v.* = f16_data[i];
                }
            },
            .Q4_0 => {
                const blocks: [*]const Q4Block = @ptrCast(@alignCast(data.ptr));
                const n_blocks = expected_size / 32;
                for (0..n_blocks) |b| {
                    const block = blocks[b];
                    const d: f32 = block.d;
                    for (0..16) |i| {
                        const byte = block.qs[i];
                        buf.slice()[b * 32 + i * 2] = (@as(f32, @floatFromInt(byte & 0xF)) - 8.0) * d;
                        buf.slice()[b * 32 + i * 2 + 1] = (@as(f32, @floatFromInt(byte >> 4)) - 8.0) * d;
                    }
                }
            },
            .Q8_0 => {
                const blocks: [*]const Q8Block = @ptrCast(@alignCast(data.ptr));
                const n_blocks = expected_size / 32;
                for (0..n_blocks) |b| {
                    const block = blocks[b];
                    const d: f32 = block.d;
                    for (0..32) |i| {
                        buf.slice()[b * 32 + i] = @as(f32, @floatFromInt(block.qs[i])) * d;
                    }
                }
            },
            else => return error.UnsupportedDtype,
        }

        return buf;
    }

    fn loadLayer(device: *metal.Device, g: *gguf.GGUFFile, config: LlamaConfig, layer: u32) !Layer {
        var name_buf: [128]u8 = undefined;

        var name = try std.fmt.bufPrint(&name_buf, "blk.{}.attn_norm.weight", .{layer});
        const attn_norm = try loadF32Tensor(device, g, name, config.dim);

        name = try std.fmt.bufPrint(&name_buf, "blk.{}.attn_q.weight", .{layer});
        const wq = try loadF32Tensor(device, g, name, config.dim * config.dim);

        name = try std.fmt.bufPrint(&name_buf, "blk.{}.attn_k.weight", .{layer});
        const wk = try loadF32Tensor(device, g, name, config.dim * config.kv_dim);

        name = try std.fmt.bufPrint(&name_buf, "blk.{}.attn_v.weight", .{layer});
        const wv = try loadF32Tensor(device, g, name, config.dim * config.kv_dim);

        name = try std.fmt.bufPrint(&name_buf, "blk.{}.attn_output.weight", .{layer});
        const wo = try loadF32Tensor(device, g, name, config.dim * config.dim);

        name = try std.fmt.bufPrint(&name_buf, "blk.{}.ffn_norm.weight", .{layer});
        const ffn_norm = try loadF32Tensor(device, g, name, config.dim);

        name = try std.fmt.bufPrint(&name_buf, "blk.{}.ffn_gate.weight", .{layer});
        const w1 = try loadF32Tensor(device, g, name, config.dim * config.hidden_dim);

        name = try std.fmt.bufPrint(&name_buf, "blk.{}.ffn_down.weight", .{layer});
        const w2 = try loadF32Tensor(device, g, name, config.hidden_dim * config.dim);

        name = try std.fmt.bufPrint(&name_buf, "blk.{}.ffn_up.weight", .{layer});
        const w3 = try loadF32Tensor(device, g, name, config.dim * config.hidden_dim);

        return .{
            .attn_norm = attn_norm,
            .wq = wq,
            .wk = wk,
            .wv = wv,
            .wo = wo,
            .ffn_norm = ffn_norm,
            .w1 = w1,
            .w2 = w2,
            .w3 = w3,
        };
    }
};

fn precomputeRoPE(cos: []f32, sin: []f32, head_dim: u32, max_seq: u32, theta: f32) void {
    const half_dim = head_dim / 2;
    for (0..max_seq) |pos| {
        for (0..half_dim) |i| {
            const freq = 1.0 / std.math.pow(f32, theta, @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(head_dim)));
            const angle = @as(f32, @floatFromInt(pos)) * freq;
            cos[pos * half_dim + i] = @cos(angle);
            sin[pos * half_dim + i] = @sin(angle);
        }
    }
}
