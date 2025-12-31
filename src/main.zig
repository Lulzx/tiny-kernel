pub const metal = @import("metal/metal.zig");
pub const kernels = @import("kernels/kernels.zig");
pub const model = @import("model/model.zig");

pub const Device = metal.Device;
pub const CommandQueue = metal.CommandQueue;
pub const Buffer = metal.Buffer;
pub const ComputePipeline = metal.ComputePipeline;

pub const rmsnorm = kernels.rmsnorm;
pub const rope = kernels.rope;
pub const swiglu = kernels.swiglu;
pub const fused = kernels.fused;
pub const attention = kernels.attention;
pub const quantized = kernels.quantized;

pub fn init() !*Device {
    return Device.init();
}

pub fn deinit(device: *Device) void {
    device.deinit();
}

test "basic init" {
    const device = try init();
    defer deinit(device);
}
