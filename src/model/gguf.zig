const std = @import("std");

pub const GGMLType = enum(u32) {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    _,
};

pub const GGUFType = enum(u32) {
    UINT8 = 0,
    INT8 = 1,
    UINT16 = 2,
    INT16 = 3,
    UINT32 = 4,
    INT32 = 5,
    FLOAT32 = 6,
    BOOL = 7,
    STRING = 8,
    ARRAY = 9,
    UINT64 = 10,
    INT64 = 11,
    FLOAT64 = 12,
    _,
};

pub const TensorInfo = struct {
    name: []const u8,
    n_dims: u32,
    dims: [4]u64,
    dtype: GGMLType,
    offset: u64,

    pub fn nelements(self: TensorInfo) u64 {
        var n: u64 = 1;
        for (0..self.n_dims) |i| {
            n *= self.dims[i];
        }
        return n;
    }

    pub fn nbytes(self: TensorInfo) u64 {
        const ne = self.nelements();
        return switch (self.dtype) {
            .F32 => ne * 4,
            .F16 => ne * 2,
            .Q4_0 => ne / 32 * 18,
            .Q4_1 => ne / 32 * 20,
            .Q8_0 => ne / 32 * 34,
            .Q4_K => ne / 256 * 144,
            .Q5_K => ne / 256 * 176,
            .Q6_K => ne / 256 * 210,
            else => ne,
        };
    }
};

pub const MetaValue = union(enum) {
    uint8: u8,
    int8: i8,
    uint16: u16,
    int16: i16,
    uint32: u32,
    int32: i32,
    float32: f32,
    bool_val: bool,
    string: []const u8,
    array: []MetaValue,
    uint64: u64,
    int64: i64,
    float64: f64,
};

pub const GGUFFile = struct {
    allocator: std.mem.Allocator,
    data: []align(std.heap.page_size_min) const u8,
    file: std.fs.File,
    version: u32,
    tensor_count: u64,
    metadata: std.StringHashMap(MetaValue),
    tensors: std.StringHashMap(TensorInfo),
    tensor_data_start: u64,

    pub fn open(allocator: std.mem.Allocator, path: []const u8) !*GGUFFile {
        const file = try std.fs.cwd().openFile(path, .{});
        errdefer file.close();

        const stat = try file.stat();
        const data = try std.posix.mmap(
            null,
            stat.size,
            std.posix.PROT.READ,
            .{ .TYPE = .PRIVATE },
            file.handle,
            0,
        );

        var self = try allocator.create(GGUFFile);
        self.* = .{
            .allocator = allocator,
            .data = data,
            .file = file,
            .version = 0,
            .tensor_count = 0,
            .metadata = std.StringHashMap(MetaValue).init(allocator),
            .tensors = std.StringHashMap(TensorInfo).init(allocator),
            .tensor_data_start = 0,
        };

        try self.parse();
        return self;
    }

    pub fn close(self: *GGUFFile) void {
        // Free all allocated arrays in metadata
        var it = self.metadata.valueIterator();
        while (it.next()) |val| {
            if (val.* == .array) {
                self.allocator.free(val.array);
            }
        }
        std.posix.munmap(self.data);
        self.file.close();
        self.metadata.deinit();
        self.tensors.deinit();
        self.allocator.destroy(self);
    }

    fn parse(self: *GGUFFile) !void {
        var pos: usize = 0;

        const magic = self.data[0..4];
        if (!std.mem.eql(u8, magic, "GGUF")) {
            return error.InvalidMagic;
        }
        pos += 4;

        self.version = std.mem.readInt(u32, self.data[pos..][0..4], .little);
        pos += 4;

        self.tensor_count = std.mem.readInt(u64, self.data[pos..][0..8], .little);
        pos += 8;

        const metadata_kv_count = std.mem.readInt(u64, self.data[pos..][0..8], .little);
        pos += 8;

        for (0..metadata_kv_count) |_| {
            const key_len = std.mem.readInt(u64, self.data[pos..][0..8], .little);
            pos += 8;
            const key = self.data[pos..][0..key_len];
            pos += key_len;

            const value_type: GGUFType = @enumFromInt(std.mem.readInt(u32, self.data[pos..][0..4], .little));
            pos += 4;

            const value = try self.readMetaValue(value_type, &pos);
            try self.metadata.put(key, value);
        }

        for (0..self.tensor_count) |_| {
            const name_len = std.mem.readInt(u64, self.data[pos..][0..8], .little);
            pos += 8;
            const name = self.data[pos..][0..name_len];
            pos += name_len;

            const n_dims = std.mem.readInt(u32, self.data[pos..][0..4], .little);
            pos += 4;

            var dims: [4]u64 = .{ 1, 1, 1, 1 };
            for (0..n_dims) |i| {
                dims[i] = std.mem.readInt(u64, self.data[pos..][0..8], .little);
                pos += 8;
            }

            const dtype: GGMLType = @enumFromInt(std.mem.readInt(u32, self.data[pos..][0..4], .little));
            pos += 4;

            const offset = std.mem.readInt(u64, self.data[pos..][0..8], .little);
            pos += 8;

            try self.tensors.put(name, .{
                .name = name,
                .n_dims = n_dims,
                .dims = dims,
                .dtype = dtype,
                .offset = offset,
            });
        }

        self.tensor_data_start = (pos + 31) & ~@as(usize, 31);
    }

    fn readMetaValue(self: *GGUFFile, value_type: GGUFType, pos: *usize) !MetaValue {
        switch (value_type) {
            .UINT8 => {
                const v = self.data[pos.*];
                pos.* += 1;
                return .{ .uint8 = v };
            },
            .INT8 => {
                const v: i8 = @bitCast(self.data[pos.*]);
                pos.* += 1;
                return .{ .int8 = v };
            },
            .UINT16 => {
                const v = std.mem.readInt(u16, self.data[pos.*..][0..2], .little);
                pos.* += 2;
                return .{ .uint16 = v };
            },
            .INT16 => {
                const v = std.mem.readInt(i16, self.data[pos.*..][0..2], .little);
                pos.* += 2;
                return .{ .int16 = v };
            },
            .UINT32 => {
                const v = std.mem.readInt(u32, self.data[pos.*..][0..4], .little);
                pos.* += 4;
                return .{ .uint32 = v };
            },
            .INT32 => {
                const v = std.mem.readInt(i32, self.data[pos.*..][0..4], .little);
                pos.* += 4;
                return .{ .int32 = v };
            },
            .FLOAT32 => {
                const bits = std.mem.readInt(u32, self.data[pos.*..][0..4], .little);
                pos.* += 4;
                return .{ .float32 = @bitCast(bits) };
            },
            .BOOL => {
                const v = self.data[pos.*] != 0;
                pos.* += 1;
                return .{ .bool_val = v };
            },
            .STRING => {
                const len = std.mem.readInt(u64, self.data[pos.*..][0..8], .little);
                pos.* += 8;
                const s = self.data[pos.*..][0..len];
                pos.* += len;
                return .{ .string = s };
            },
            .ARRAY => {
                const elem_type: GGUFType = @enumFromInt(std.mem.readInt(u32, self.data[pos.*..][0..4], .little));
                pos.* += 4;
                const len = std.mem.readInt(u64, self.data[pos.*..][0..8], .little);
                pos.* += 8;
                var arr = try self.allocator.alloc(MetaValue, len);
                for (0..len) |i| {
                    arr[i] = try self.readMetaValue(elem_type, pos);
                }
                return .{ .array = arr };
            },
            .UINT64 => {
                const v = std.mem.readInt(u64, self.data[pos.*..][0..8], .little);
                pos.* += 8;
                return .{ .uint64 = v };
            },
            .INT64 => {
                const v = std.mem.readInt(i64, self.data[pos.*..][0..8], .little);
                pos.* += 8;
                return .{ .int64 = v };
            },
            .FLOAT64 => {
                const bits = std.mem.readInt(u64, self.data[pos.*..][0..8], .little);
                pos.* += 8;
                return .{ .float64 = @bitCast(bits) };
            },
            _ => return error.UnknownMetaType,
        }
    }

    pub fn getTensor(self: *GGUFFile, name: []const u8) ?TensorInfo {
        return self.tensors.get(name);
    }

    pub fn getTensorData(self: *GGUFFile, info: TensorInfo) []const u8 {
        const start = self.tensor_data_start + info.offset;
        const end = start + info.nbytes();
        return self.data[start..end];
    }

    pub fn getMetaU32(self: *GGUFFile, key: []const u8) ?u32 {
        const val = self.metadata.get(key) orelse return null;
        return switch (val) {
            .uint32 => |v| v,
            .int32 => |v| @intCast(v),
            .uint64 => |v| @intCast(v),
            else => null,
        };
    }

    pub fn getMetaF32(self: *GGUFFile, key: []const u8) ?f32 {
        const val = self.metadata.get(key) orelse return null;
        return switch (val) {
            .float32 => |v| v,
            .float64 => |v| @floatCast(v),
            else => null,
        };
    }

    pub fn getMetaString(self: *GGUFFile, key: []const u8) ?[]const u8 {
        const val = self.metadata.get(key) orelse return null;
        return switch (val) {
            .string => |v| v,
            else => null,
        };
    }

    pub fn getMetaArray(self: *GGUFFile, key: []const u8) ?[]MetaValue {
        const val = self.metadata.get(key) orelse return null;
        return switch (val) {
            .array => |v| v,
            else => null,
        };
    }
};
