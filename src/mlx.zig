const std = @import("std");
const mlx = @cImport({
    @cInclude("mlx.h");
    @cInclude("mlx_types.h");
});

pub usingnamespace mlx;
pub usingnamespace @import("array.zig");
pub const ops = @import("ops.zig");

pub inline fn MLX_CHECK(v: mlx.mlx_err, src: std.builtin.SourceLocation) !void {
    if (v != mlx.mlx_success) {
        std.debug.print("MLX Exception - {s}:{d}\n", .{ src.file, src.line });
        return error.MLXThrewException;
    }
}

test {
    _ = ops;
}

test "MLX -> seed" {
    try MLX_CHECK(mlx.seed(12345), @src());
}

test "MLX -> randomNormal" {
    const shape: []const c_int = &.{ 4, 4, 4 };
    var arr: mlx.mlx_array = null;
    try MLX_CHECK(mlx.randomNormal(&arr, shape.ptr, shape.len, mlx.float32), @src());
    defer mlx.destroyArray(arr);
}

test "MLX -> fromSlice" {
    const shape: []const c_int = &.{10};
    const data: []const f32 = &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    var arr: mlx.mlx_array = null;
    try MLX_CHECK(mlx.fromPtr(&arr, data.ptr, shape.ptr, shape.len, mlx.float32), @src());
    defer mlx.destroyArray(arr);
    var ptr: ?*anyopaque = null;
    try MLX_CHECK(mlx.data(&ptr, arr), @src());
    const array_data: []f32 = @as([*c]f32, @ptrCast(@alignCast(ptr)))[0..10];
    try std.testing.expectEqualSlices(f32, data, array_data);
}

test "MLX -> dtype" {
    const shape: []const c_int = &.{10};
    const data: []const i8 = &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    var arr: mlx.mlx_array = null;
    try MLX_CHECK(mlx.fromPtr(&arr, data.ptr, shape.ptr, shape.len, mlx.int8), @src());
    defer mlx.destroyArray(arr);
    var dtype: mlx.mlx_dtype = undefined;
    try MLX_CHECK(mlx.dtype(&dtype, arr), @src());
    try std.testing.expectEqual(dtype, mlx.int8);
}
