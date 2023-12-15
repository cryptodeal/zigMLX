const std = @import("std");
const mlx = @import("mlx.zig");

/// Convenience wrapper around ptr to MLX's array.
pub const Array = struct {
    /// Pointer to the underlying MLX array.
    handle: mlx.mlx_array = null,

    pub fn init(handle: mlx.mlx_array) Array {
        return .{ .handle = handle };
    }

    /// Initialize an MLX array from an `f64` scalar value.
    pub fn fromScalar(val: f64, data_type: mlx.mlx_dtype) !Array {
        var handle: mlx.mlx_array = null;
        try mlx.MLX_CHECK(mlx.fromScalar(&handle, val, data_type), @src());
        return .{ .handle = handle };
    }

    /// Initialize an MLX array from an `i64` scalar value.
    pub fn fromScalarI64(val: i64) !Array {
        var handle: mlx.mlx_array = null;
        try mlx.MLX_CHECK(mlx.fromScalarI64(&handle, val), @src());
        return .{ .handle = handle };
    }

    /// Initialize an MLX array from an `u64` scalar value.
    pub fn fromScalarU64(val: u64) !Array {
        var handle: mlx.mlx_array = null;
        try mlx.MLX_CHECK(mlx.fromScalarI64(&handle, val), @src());
        return .{ .handle = handle };
    }

    /// Initialize an MLX array of the given shape and dtype.
    pub fn initHandle(shape_: []const i64, data_type: mlx.mlx_dtype) !Array {
        var handle: mlx.mlx_array = null;
        try mlx.MLX_CHECK(mlx.initHandle(&handle, shape_.ptr, @intCast(shape_.len), data_type), @src());
        return .{ .handle = handle };
    }

    /// Initialize an empty MLX array.
    pub fn initEmpty() !Array {
        var handle: mlx.mlx_array = null;
        try mlx.MLX_CHECK(mlx.initEmpty(&handle), @src());
        return .{ .handle = handle };
    }

    /// Initialize an MLX array from a slice.
    pub fn fromSlice(comptime T: type, d: []const T, shape_: []const i64, data_type: mlx.mlx_dtype) !Array {
        var handle: mlx.mlx_array = null;
        try mlx.MLX_CHECK(mlx.fromPtr(&handle, d.ptr, shape_.ptr, @intCast(shape_.len), data_type), @src());
        return .{ .handle = handle };
    }

    /// Intialize a random MLX array with normal distribution of the given
    /// shape and dtype.
    pub fn randomNormal(shape_: []const i64, data_type: mlx.mlx_dtype) !Array {
        var handle: mlx.mlx_array = null;
        try mlx.MLX_CHECK(mlx.randomNormal(&handle, shape_.ptr, @intCast(shape_.len), data_type), @src());
        return .{ .handle = handle };
    }

    /// Frees the underlying memory of the MLX array.
    pub fn deinit(self: *Array) void {
        if (self.handle != null) {
            mlx.destroyArray(self.handle);
            self.handle = null;
        }
    }

    /// Returns the size of the MLX array's datatype in bytes.
    pub fn itemsize(self: *const Array) !usize {
        var res: usize = undefined;
        try mlx.MLX_CHECK(mlx.itemsize(&res, self.handle), @src());
        return res;
    }

    /// Returns the number of elements in the MLX array.
    pub fn size(self: *const Array) !usize {
        var res: usize = undefined;
        try mlx.MLX_CHECK(mlx.size(&res, self.handle), @src());
        return res;
    }

    /// Returns the number of bytes in the MLX array.
    pub fn nbytes(self: *const Array) !usize {
        var res: usize = undefined;
        try mlx.MLX_CHECK(mlx.nbytes(&res, self.handle), @src());
        return res;
    }

    /// Returns the number of dimensions of the array.
    pub fn ndim(self: *const Array) !usize {
        var res: usize = undefined;
        try mlx.MLX_CHECK(mlx.ndim(&res, self.handle), @src());
        return res;
    }

    /// Returns the shape of the MLX array.
    pub fn shape(self: *const Array, allocator: std.mem.Allocator) ![]i64 {
        const ndims = try self.ndim();
        var c_shape: ?*anyopaque = null;
        try mlx.MLX_CHECK(mlx.shape(&c_shape, self.handle), @src());
        var res = try allocator.alloc(i64, ndims);
        for (@as([*c]c_int, @ptrCast(@alignCast(c_shape)))[0..ndims], 0..) |v, i| {
            res[i] = @intCast(v);
        }
        return res;
    }

    /// Returns the size of the corresponding dimension of the MLX array.
    ///
    /// This function supports negative indexing and provides
    /// bounds checking.
    pub fn dim(self: *const Array, dimension: i64) !i64 {
        var res: c_int = undefined;
        try mlx.MLX_CHECK(mlx.dim(&res, @intCast(dimension), self.handle), @src());
        return @intCast(res);
    }

    /// Returns the strides of the MLX array.
    pub fn strides(self: *const Array, allocator: std.mem.Allocator) ![]usize {
        var stride_len: usize = undefined;
        var c_strides: ?*anyopaque = null;
        try mlx.MLX_CHECK(mlx.strides(&c_strides, &stride_len, self.handle), @src());
        const res = try allocator.alloc(usize, stride_len);
        @memcpy(res, @as([*c]usize, @ptrCast(@alignCast(c_strides)))[0..stride_len]);
        return res;
    }

    /// Returns the data type of the MLX array.
    pub fn dtype(self: *const Array) !mlx.mlx_dtype {
        var data_type: mlx.mlx_dtype = undefined;
        try mlx.MLX_CHECK(mlx.dtype(&data_type, self.handle), @src());
        return data_type;
    }

    /// Evaluates the MLX array.
    pub fn eval(self: *const Array, retain_graph: bool) !void {
        return mlx.MLX_CHECK(mlx.eval_array(retain_graph, self.handle), @src());
    }

    /// Returns the value from a scalar array
    pub fn item(self: *const Array, comptime T: type, retain_graph: bool) !T {
        var res: T = undefined;
        try mlx.MLX_CHECK(mlx.item(&res, retain_graph, self.handle), @src());
        return res;
    }

    pub fn begin(self: *const Array) !ArrayIterator {
        var iter: mlx.mlx_array_iterator = null;
        try mlx.MLX_CHECK(mlx.begin(&iter, self.handle), @src());
        return .{ .handle = iter };
    }

    pub fn end(self: *const Array) !ArrayIterator {
        var iter: mlx.mlx_array_iterator = null;
        try mlx.MLX_CHECK(mlx.end(&iter, self.handle), @src());
        return .{ .handle = iter };
    }

    /// Returns a unique identifier for the MLX array.
    pub fn id(self: *const Array) !usize {
        var res: usize = undefined;
        try mlx.MLX_CHECK(mlx.id(&res, self.handle), @src());
        return res;
    }

    /// Returns a pointer to the MLX array's primitive.
    pub fn primitive(self: *const Array) !mlx.mlx_primitive {
        var prim: mlx.mlx_primitive = null;
        try mlx.MLX_CHECK(mlx.primitive(&prim, self.handle), @src());
        return prim;
    }

    /// Check if the MLX array has an attached primitive or is a leaf node.
    pub fn hasPrimitive(self: *const Array) !bool {
        var res: bool = undefined;
        try mlx.MLX_CHECK(mlx.has_primitive(&res, self.handle), @src());
        return res;
    }

    // TODO: pub fn inputs() ![]const Array {}

    // TODO: pub fn editableInputs() ![]Array {}

    /// Detaches the MLX array from the graph.
    pub fn detach(self: *const Array) !void {
        return mlx.MLX_CHECK(mlx.detach(self.handle), @src());
    }

    pub fn flags(self: *const Array) !mlx.mlx_array_flags {
        var res: mlx.mlx_array_flags = undefined;
        try mlx.MLX_CHECK(mlx.flags(&res, self.handle), @src());
        return res;
    }

    /// The size (in elements) of the underlying buffer the MLX array points to.
    pub fn dataSize(self: *const Array) !usize {
        var res: usize = undefined;
        try mlx.MLX_CHECK(mlx.data_size(&res, self.handle), @src());
        return res;
    }

    /// Returns slice (non-allocated) to the underlying data of the MLX array.
    pub fn data(self: *const Array, comptime T: type) ![]const T {
        var ptr: ?*anyopaque = null;
        try mlx.MLX_CHECK(mlx.data(&ptr, self.handle), @src());
        return @as([*c]T, @ptrCast(@alignCast(ptr)))[0..try self.size()];
    }

    /// Returns slice (allocated) to the underlying data of the MLX array.
    pub fn allocData(self: *const Array, comptime T: type, allocator: std.mem.Allocator) ![]T {
        var ptr: ?*anyopaque = null;
        try mlx.MLX_CHECK(mlx.data(&ptr, self.handle), @src());
        const res = try allocator.alloc(T, try self.size());
        @memcpy(res, @as([*c]T, @ptrCast(@alignCast(ptr)))[0..try self.size()]);
        return res;
    }

    /// Check if the MLX array has been evaluated.
    pub fn isEvaled(self: *const Array) !bool {
        var res: bool = undefined;
        try mlx.MLX_CHECK(mlx.is_evaled(&res, self.handle), @src());
        return res;
    }

    /// Mark the MLX array as a tracer array (true) or not.
    pub fn setTracer(self: *const Array, is_tracer: bool) !void {
        return mlx.MLX_CHECK(mlx.set_tracer(is_tracer, self.handle), @src());
    }
};

pub const ArrayIterator = struct {
    handle: mlx.mlx_array_iterator = null,

    pub fn deinit(self: *ArrayIterator) void {
        if (self.handle != null) {
            mlx.destroyArrayIterator(self.handle);
            self.handle = null;
        }
    }

    pub fn nextDiff(self: *const ArrayIterator, diff: usize) !void {
        return mlx.MLX_CHECK(mlx.nextDiff(self.handle, diff), @src());
    }

    pub fn next(self: *const ArrayIterator) !void {
        return mlx.MLX_CHECK(mlx.next(self.handle), @src());
    }

    pub fn eql(self: *const ArrayIterator, other: ArrayIterator) !bool {
        var res: bool = undefined;
        try mlx.MLX_CHECK(mlx.eql(&res, self.handle, other.handle), @src());
        return res;
    }

    pub fn neq(self: *const ArrayIterator, other: ArrayIterator) !bool {
        var res: bool = undefined;
        try mlx.MLX_CHECK(mlx.neql(&res, self.handle, other.handle), @src());
        return res;
    }
};

test "Array -> fromScalar" {
    var arr = try Array.fromScalar(10, mlx.float32);
    defer arr.deinit();
    try std.testing.expect(try arr.size() == 1);
    try std.testing.expect(try arr.dtype() == mlx.float32);
    try std.testing.expect(try arr.item(f32, false) == 10);
}

test "Array -> Flags" {
    const shape: []const c_int = &.{10};
    const data: []const f32 = &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    var arr: mlx.mlx_array = null;
    try mlx.MLX_CHECK(mlx.fromPtr(&arr, data.ptr, shape.ptr, shape.len, mlx.float32), @src());
    var wrapped = Array{ .handle = arr };
    defer wrapped.deinit(); // wrapping array manually created, so only free once
    const flags = try wrapped.flags();
    try std.testing.expect(flags.contiguous);
    try std.testing.expect(flags.row_contiguous);
    try std.testing.expect(flags.col_contiguous);
}
