const mlx = @import("mlx.zig");
const std = @import("std");

const Array = mlx.Array;

const OperatorLoc = enum { Lhs, Rhs };

fn createTmpScalar(comptime T: type, val: T, dtype: mlx.mlx_dtype, comptime fn_name: [:0]const u8, comptime side: OperatorLoc) !Array {
    const TypeInfo = @typeInfo(T);
    if (TypeInfo != .Int and TypeInfo != .Float) {
        @compileError(fn_name ++ ": Invalid Type passed for " ++ @tagName(side));
    }
    var res: mlx.mlx_array = null;
    if (TypeInfo == .Int) {
        switch (T) {
            .u64 => try mlx.MLX_CHECK(mlx.fromScalarU64(&res, val), @src()),
            .i64 => try mlx.MLX_CHECK(mlx.fromScalarI64(&res, val), @src()),
            else => try mlx.MLX_CHECK(mlx.fromScalar(&res, @floatFromInt(val), dtype), @src()),
        }
    } else if (TypeInfo == .Float) {
        try mlx.MLX_CHECK(mlx.fromScalar(&res, @floatCast(val), dtype), @src());
    }
    return Array.init(res);
}

pub fn add(comptime LhsT: type, lhs: LhsT, comptime RhsT: type, rhs: RhsT) !Array {
    if (LhsT != Array and RhsT != Array) {
        @compileError("add: at least one of the arguments must be an Array");
    }
    const fn_name = @src().fn_name;
    var res: mlx.mlx_array = null;
    if (LhsT == Array and RhsT == Array) {
        try mlx.MLX_CHECK(mlx.add(&res, lhs.handle, rhs.handle), @src());
    } else if (LhsT == Array) {
        var tmp = try createTmpScalar(RhsT, rhs, try lhs.dtype(), fn_name, .Rhs);
        defer tmp.deinit();
        try mlx.MLX_CHECK(mlx.add(&res, lhs.handle, tmp.handle), @src());
    } else {
        var tmp = try createTmpScalar(LhsT, lhs, try rhs.dtype(), fn_name, .Lhs);
        defer tmp.deinit();
        try mlx.MLX_CHECK(mlx.add(&res, tmp.handle, rhs.handle), @src());
    }
    return Array.init(res);
}

pub fn subtract(comptime LhsT: type, lhs: LhsT, comptime RhsT: type, rhs: RhsT) !Array {
    if (LhsT != Array and RhsT != Array) {
        @compileError("subtract: at least one of the arguments must be an Array");
    }
    const fn_name = @src().fn_name;
    var res: mlx.mlx_array = null;
    if (LhsT == Array and RhsT == Array) {
        try mlx.MLX_CHECK(mlx.subtract(&res, lhs.handle, rhs.handle), @src());
    } else if (LhsT == Array) {
        var tmp = try createTmpScalar(RhsT, rhs, try lhs.dtype(), fn_name, .Rhs);
        defer tmp.deinit();
        try mlx.MLX_CHECK(mlx.subtract(&res, lhs.handle, tmp.handle), @src());
    } else {
        var tmp = try createTmpScalar(LhsT, lhs, try rhs.dtype(), fn_name, .Lhs);
        defer tmp.deinit();
        try mlx.MLX_CHECK(mlx.subtract(&res, tmp.handle, rhs.handle), @src());
    }
    return Array.init(res);
}

pub fn multiply(comptime LhsT: type, lhs: LhsT, comptime RhsT: type, rhs: RhsT) !Array {
    if (LhsT != Array and RhsT != Array) {
        @compileError("multiply: at least one of the arguments must be an Array");
    }
    const fn_name = @src().fn_name;
    var res: mlx.mlx_array = null;
    if (LhsT == Array and RhsT == Array) {
        try mlx.MLX_CHECK(mlx.multiply(&res, lhs.handle, rhs.handle), @src());
    } else if (LhsT == Array) {
        var tmp = try createTmpScalar(RhsT, rhs, try lhs.dtype(), fn_name, .Rhs);
        defer tmp.deinit();
        try mlx.MLX_CHECK(mlx.multiply(&res, lhs.handle, tmp.handle), @src());
    } else {
        var tmp = try createTmpScalar(LhsT, lhs, try rhs.dtype(), fn_name, .Lhs);
        defer tmp.deinit();
        try mlx.MLX_CHECK(mlx.multiply(&res, tmp.handle, rhs.handle), @src());
    }
    return Array.init(res);
}

pub fn divide(comptime LhsT: type, lhs: LhsT, comptime RhsT: type, rhs: RhsT) !Array {
    if (LhsT != Array and RhsT != Array) {
        @compileError("divide: at least one of the arguments must be an Array");
    }
    const fn_name = @src().fn_name;
    var res: mlx.mlx_array = null;
    if (LhsT == Array and RhsT == Array) {
        try mlx.MLX_CHECK(mlx.divide(&res, lhs.handle, rhs.handle), @src());
    } else if (LhsT == Array) {
        var tmp = try createTmpScalar(RhsT, rhs, try lhs.dtype(), fn_name, .Rhs);
        defer tmp.deinit();
        try mlx.MLX_CHECK(mlx.divide(&res, lhs.handle, tmp.handle), @src());
    } else {
        var tmp = try createTmpScalar(LhsT, lhs, try rhs.dtype(), fn_name, .Lhs);
        defer tmp.deinit();
        try mlx.MLX_CHECK(mlx.divide(&res, tmp.handle, rhs.handle), @src());
    }
    return Array.init(res);
}

test "Ops -> add" {
    var a = try Array.fromSlice(f32, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }, &.{10}, mlx.float32);
    defer a.deinit();
    var b = try Array.fromSlice(f32, &.{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }, &.{10}, mlx.float32);
    defer b.deinit();
    var c = try add(Array, a, Array, b);
    defer c.deinit();
    try c.eval(false);
    const c_data = try c.data(f32);
    try std.testing.expectEqualSlices(f32, c_data, &.{ 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 });

    var d = try add(Array, a, f32, 2);
    defer d.deinit();
    try d.eval(false);
    const d_data = try d.data(f32);
    try std.testing.expectEqualSlices(f32, d_data, &.{ 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });

    var e = try add(f32, 3, Array, a);
    defer e.deinit();
    try e.eval(false);
    const e_data = try e.data(f32);
    try std.testing.expectEqualSlices(f32, e_data, &.{ 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 });
}

test "Ops -> subtract" {
    var a = try Array.fromSlice(f32, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }, &.{10}, mlx.float32);
    defer a.deinit();
    var b = try Array.fromSlice(f32, &.{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }, &.{10}, mlx.float32);
    defer b.deinit();
    var c = try subtract(Array, a, Array, b);
    defer c.deinit();
    try c.eval(false);
    const c_data = try c.data(f32);
    try std.testing.expectEqualSlices(f32, c_data, &.{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 });

    var d = try subtract(Array, a, f32, 2);
    defer d.deinit();
    try d.eval(false);
    const d_data = try d.data(f32);
    try std.testing.expectEqualSlices(f32, d_data, &.{ -1, 0, 1, 2, 3, 4, 5, 6, 7, 8 });

    var e = try subtract(f32, 3, Array, a);
    defer e.deinit();
    try e.eval(false);
    const e_data = try e.data(f32);
    try std.testing.expectEqualSlices(f32, e_data, &.{ 2, 1, 0, -1, -2, -3, -4, -5, -6, -7 });
}
