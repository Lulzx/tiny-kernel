const std = @import("std");
const tk = @import("main.zig");
const model = @import("model/model.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 3) {
        std.debug.print("Usage: {s} <model.gguf> \"<prompt>\" [max_tokens] [temperature]\n", .{args[0]});
        std.debug.print("\nExample:\n", .{});
        std.debug.print("  {s} tinyllama.gguf \"Once upon a time\" 64 0.7\n", .{args[0]});
        return;
    }

    const model_path = args[1];
    const prompt = args[2];
    const max_tokens: usize = if (args.len > 3) try std.fmt.parseInt(usize, args[3], 10) else 64;
    const temperature: f32 = if (args.len > 4) try std.fmt.parseFloat(f32, args[4]) else 0.8;

    std.debug.print("tiny-kernel inference\n", .{});
    std.debug.print("=====================\n\n", .{});

    const device = try tk.init();
    defer tk.deinit(device);
    std.debug.print("Device: {s}\n\n", .{device.name()});

    std.debug.print("Loading model: {s}\n", .{model_path});
    var llama = try model.Llama.init(allocator, device, model_path);
    defer llama.deinit();

    std.debug.print("\nLoading tokenizer...\n", .{});
    var tokenizer = try model.Tokenizer.init(allocator, llama.gguf_file);
    defer tokenizer.deinit();
    std.debug.print("Vocab size: {}\n\n", .{tokenizer.vocabSize()});

    std.debug.print("Encoding prompt: \"{s}\"\n", .{prompt});
    const prompt_tokens = try tokenizer.encode(prompt, false);
    defer allocator.free(prompt_tokens);
    std.debug.print("Prompt tokens: {}\n\n", .{prompt_tokens.len});

    std.debug.print("Generating (max_tokens={}, temperature={d:.2})...\n", .{ max_tokens, temperature });
    std.debug.print("---\n", .{});

    const decoded_prompt = try tokenizer.decode(prompt_tokens);
    defer allocator.free(decoded_prompt);
    std.debug.print("{s}", .{decoded_prompt});

    const start_time = std.time.nanoTimestamp();

    var pos: u32 = 0;
    for (prompt_tokens) |token| {
        try llama.forward(token, pos);
        pos += 1;
    }

    var generated: usize = 0;
    var next_token = llama.sample(temperature);

    while (generated < max_tokens) {
        if (next_token == tokenizer.eos_id) break;

        const piece = try tokenizer.decode(&[_]u32{next_token});
        defer allocator.free(piece);
        std.debug.print("{s}", .{piece});

        try llama.forward(next_token, pos);
        pos += 1;
        generated += 1;

        next_token = llama.sample(temperature);
    }

    const end_time = std.time.nanoTimestamp();
    const elapsed_ms = @as(f64, @floatFromInt(end_time - start_time)) / 1_000_000.0;
    const tokens_per_sec = @as(f64, @floatFromInt(generated)) / (elapsed_ms / 1000.0);

    std.debug.print("\n---\n", .{});
    std.debug.print("Generated {} tokens in {d:.1} ms ({d:.1} tokens/sec)\n", .{ generated, elapsed_ms, tokens_per_sec });
}
