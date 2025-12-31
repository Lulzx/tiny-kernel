const std = @import("std");
const gguf = @import("gguf.zig");

pub const Tokenizer = struct {
    allocator: std.mem.Allocator,
    vocab: [][]const u8,
    scores: []f32,
    vocab_map: std.StringHashMap(u32),
    bos_id: u32,
    eos_id: u32,

    pub fn init(allocator: std.mem.Allocator, gguf_file: *gguf.GGUFFile) !Tokenizer {
        const tokens_arr = gguf_file.getMetaArray("tokenizer.ggml.tokens") orelse return error.NoTokens;
        const scores_arr = gguf_file.getMetaArray("tokenizer.ggml.scores");

        var vocab = try allocator.alloc([]const u8, tokens_arr.len);
        var scores = try allocator.alloc(f32, tokens_arr.len);
        var vocab_map = std.StringHashMap(u32).init(allocator);

        for (tokens_arr, 0..) |tok, i| {
            vocab[i] = tok.string;
            scores[i] = if (scores_arr) |sa| sa[i].float32 else 0.0;
            try vocab_map.put(vocab[i], @intCast(i));
        }

        const bos_id = gguf_file.getMetaU32("tokenizer.ggml.bos_token_id") orelse 1;
        const eos_id = gguf_file.getMetaU32("tokenizer.ggml.eos_token_id") orelse 2;

        return .{
            .allocator = allocator,
            .vocab = vocab,
            .scores = scores,
            .vocab_map = vocab_map,
            .bos_id = bos_id,
            .eos_id = eos_id,
        };
    }

    pub fn deinit(self: *Tokenizer) void {
        self.allocator.free(self.vocab);
        self.allocator.free(self.scores);
        self.vocab_map.deinit();
    }

    pub fn encode(self: *Tokenizer, text: []const u8, add_bos: bool) ![]u32 {
        var tokens = std.ArrayListUnmanaged(u32){};
        defer tokens.deinit(self.allocator);

        if (add_bos) {
            try tokens.append(self.allocator, self.bos_id);
        }

        // Convert input text to GPT-2 byte encoding
        var encoded_text = std.ArrayListUnmanaged(u8){};
        defer encoded_text.deinit(self.allocator);

        for (text) |byte| {
            try appendGpt2Encoded(&encoded_text, self.allocator, byte);
        }

        var i: usize = 0;
        while (i < encoded_text.items.len) {
            var best_len: usize = 0;
            var best_id: u32 = 0;

            var try_len: usize = @min(encoded_text.items.len - i, 64);
            while (try_len > 0) : (try_len -= 1) {
                const substr = encoded_text.items[i .. i + try_len];
                if (self.vocab_map.get(substr)) |id| {
                    best_len = try_len;
                    best_id = id;
                    break;
                }
            }

            if (best_len == 0) {
                // Try byte fallback token
                const byte_token = try std.fmt.allocPrint(self.allocator, "<0x{X:0>2}>", .{encoded_text.items[i]});
                defer self.allocator.free(byte_token);
                if (self.vocab_map.get(byte_token)) |id| {
                    try tokens.append(self.allocator, id);
                }
                i += 1;
            } else {
                try tokens.append(self.allocator, best_id);
                i += best_len;
            }
        }

        // BPE merge loop
        while (true) {
            var best_score: f32 = -std.math.inf(f32);
            var best_idx: ?usize = null;
            var best_merged_id: u32 = 0;

            if (tokens.items.len < 2) break;

            for (0..tokens.items.len - 1) |idx| {
                const merged = try std.fmt.allocPrint(self.allocator, "{s}{s}", .{
                    self.vocab[tokens.items[idx]],
                    self.vocab[tokens.items[idx + 1]],
                });
                defer self.allocator.free(merged);

                if (self.vocab_map.get(merged)) |id| {
                    if (self.scores[id] > best_score) {
                        best_score = self.scores[id];
                        best_idx = idx;
                        best_merged_id = id;
                    }
                }
            }

            if (best_idx == null) break;

            const idx = best_idx.?;
            tokens.items[idx] = best_merged_id;
            _ = tokens.orderedRemove(idx + 1);
        }

        return try tokens.toOwnedSlice(self.allocator);
    }

    fn appendGpt2Encoded(list: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, byte: u8) !void {
        // GPT-2 byte-level BPE: certain bytes are encoded as 2-byte UTF-8
        // Bytes that stay as-is: 0x21-0x7E, 0xA1-0xAC, 0xAE-0xFF
        const is_direct = (byte >= 0x21 and byte <= 0x7E) or
            (byte >= 0xA1 and byte <= 0xAC) or
            (byte >= 0xAE);

        if (is_direct) {
            try list.append(allocator, byte);
            return;
        }

        // Map escaped bytes to UTF-8 encoded codepoints U+0100+
        // Space (0x20) is at index 32, maps to U+0120 = 0xC4 0xA0
        const escaped_bytes = [_]u8{
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
            0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F,
            0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
            0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F,
            0x20, // space at index 32
            0x7F, 0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86,
            0x87, 0x88, 0x89, 0x8A, 0x8B, 0x8C, 0x8D, 0x8E,
            0x8F, 0x90, 0x91, 0x92, 0x93, 0x94, 0x95, 0x96,
            0x97, 0x98, 0x99, 0x9A, 0x9B, 0x9C, 0x9D, 0x9E,
            0x9F, 0xA0, 0xAD,
        };

        for (escaped_bytes, 0..) |eb, idx| {
            if (byte == eb) {
                // Encode as UTF-8: U+0100 + idx
                const codepoint: u32 = 0x100 + @as(u32, @intCast(idx));
                // 2-byte UTF-8: 110xxxxx 10xxxxxx
                try list.append(allocator, @truncate(0xC0 | (codepoint >> 6)));
                try list.append(allocator, @truncate(0x80 | (codepoint & 0x3F)));
                return;
            }
        }

        // Fallback - shouldn't reach here
        try list.append(allocator, byte);
    }

    pub fn decode(self: *Tokenizer, tokens: []const u32) ![]u8 {
        var result = std.ArrayListUnmanaged(u8){};
        defer result.deinit(self.allocator);

        for (tokens) |tok| {
            if (tok < self.vocab.len) {
                const piece = self.vocab[tok];
                // Handle byte tokens like <0xFF>
                if (piece.len >= 6 and std.mem.startsWith(u8, piece, "<0x") and piece[piece.len - 1] == '>') {
                    const hex = piece[3 .. piece.len - 1];
                    if (std.fmt.parseInt(u8, hex, 16)) |byte| {
                        try result.append(self.allocator, byte);
                    } else |_| {
                        try result.appendSlice(self.allocator, piece);
                    }
                } else if (std.mem.startsWith(u8, piece, "<")) {
                    // Skip special tokens like <|endoftext|>
                } else {
                    // Decode GPT-2 style byte-level BPE
                    try decodeGpt2Bytes(&result, self.allocator, piece);
                }
            }
        }

        return try result.toOwnedSlice(self.allocator);
    }

    fn decodeGpt2Bytes(result: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, piece: []const u8) !void {
        var i: usize = 0;
        while (i < piece.len) {
            // Check for 2-byte UTF-8 sequences (U+0080-U+07FF)
            if (i + 1 < piece.len and (piece[i] & 0xE0) == 0xC0) {
                const c1 = piece[i];
                const c2 = piece[i + 1];
                if ((c2 & 0xC0) == 0x80) {
                    // Decode UTF-8 to codepoint
                    const codepoint = (@as(u32, c1 & 0x1F) << 6) | @as(u32, c2 & 0x3F);
                    // GPT-2 maps bytes 0x00-0x20, 0x7F-0xA0, 0xAD to U+0100+
                    if (codepoint >= 0x100 and codepoint <= 0x143) {
                        // Map back to original byte
                        const byte = gpt2DecodeByte(codepoint);
                        try result.append(allocator, byte);
                        i += 2;
                        continue;
                    }
                }
            }
            // Regular byte
            try result.append(allocator, piece[i]);
            i += 1;
        }
    }

    fn gpt2DecodeByte(codepoint: u32) u8 {
        // GPT-2 byte-level BPE encoding:
        // Bytes that map directly: 0x21-0x7E, 0xA1-0xAC, 0xAE-0xFF
        // Other bytes are mapped to U+0100 onwards in order:
        // 0x00->U+0100, 0x01->U+0101, ..., 0x20->U+0120 (space), ...
        const escaped_bytes = [_]u8{
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
            0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F,
            0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
            0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F,
            0x20, // space -> U+0120
            0x7F, 0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86,
            0x87, 0x88, 0x89, 0x8A, 0x8B, 0x8C, 0x8D, 0x8E,
            0x8F, 0x90, 0x91, 0x92, 0x93, 0x94, 0x95, 0x96,
            0x97, 0x98, 0x99, 0x9A, 0x9B, 0x9C, 0x9D, 0x9E,
            0x9F, 0xA0, 0xAD,
        };
        const idx = codepoint - 0x100;
        if (idx < escaped_bytes.len) {
            return escaped_bytes[idx];
        }
        return @truncate(codepoint);
    }

    pub fn vocabSize(self: *Tokenizer) u32 {
        return @intCast(self.vocab.len);
    }
};
