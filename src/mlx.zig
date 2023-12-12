const std = @import("std");
const mlx = @cImport({
    @cInclude("mlx.h");
    @cInclude("mlx_types.h");
});

pub usingnamespace mlx;

pub inline fn MLX_CHECK(v: mlx.mlx_err, src: std.builtin.SourceLocation) !void {
    if (v != mlx.mlx_success) {
        std.debug.print("MLX Exception - {s}:{d}\n", .{ src.file, src.line });
        return error.MLXThrewException;
    }
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
