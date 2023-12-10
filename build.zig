const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const lib = b.addStaticLibrary(.{
        .name = "zigMLX",
        .root_source_file = .{ .path = "bindings/mlx.cc" },
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    lib.linkLibCpp();
    b.installArtifact(lib);
}
