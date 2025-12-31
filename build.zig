const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const lib_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    lib_mod.linkFramework("Metal", .{});
    lib_mod.linkFramework("Foundation", .{});
    lib_mod.linkFramework("CoreGraphics", .{});
    lib_mod.linkSystemLibrary("c", .{});

    const lib = b.addLibrary(.{
        .linkage = .static,
        .name = "tiny-kernel",
        .root_module = lib_mod,
    });

    b.installArtifact(lib);

    const exe_mod = b.createModule(.{
        .root_source_file = b.path("src/bench.zig"),
        .target = target,
        .optimize = optimize,
    });

    exe_mod.linkFramework("Metal", .{});
    exe_mod.linkFramework("Foundation", .{});
    exe_mod.linkFramework("CoreGraphics", .{});
    exe_mod.linkSystemLibrary("c", .{});

    const exe = b.addExecutable(.{
        .name = "tiny-kernel-bench",
        .root_module = exe_mod,
    });

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run benchmarks");
    run_step.dependOn(&run_cmd.step);

    const gen_mod = b.createModule(.{
        .root_source_file = b.path("src/generate.zig"),
        .target = target,
        .optimize = optimize,
    });

    gen_mod.linkFramework("Metal", .{});
    gen_mod.linkFramework("Foundation", .{});
    gen_mod.linkFramework("CoreGraphics", .{});
    gen_mod.linkSystemLibrary("c", .{});

    const gen_exe = b.addExecutable(.{
        .name = "tiny-generate",
        .root_module = gen_mod,
    });

    b.installArtifact(gen_exe);

    const gen_run_cmd = b.addRunArtifact(gen_exe);
    gen_run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        gen_run_cmd.addArgs(args);
    }

    const gen_step = b.step("generate", "Run text generation");
    gen_step.dependOn(&gen_run_cmd.step);

    const test_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    test_mod.linkFramework("Metal", .{});
    test_mod.linkFramework("Foundation", .{});
    test_mod.linkFramework("CoreGraphics", .{});
    test_mod.linkSystemLibrary("c", .{});

    const lib_tests = b.addTest(.{
        .root_module = test_mod,
    });

    const run_lib_tests = b.addRunArtifact(lib_tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_lib_tests.step);
}
