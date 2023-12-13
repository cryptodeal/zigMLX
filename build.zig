const std = @import("std");

fn addLibInfo(allocator: std.mem.Allocator, c: *std.Build.Step.Compile, path_info: []const u8) !void {
    const dir_path = std.fs.path.dirname(path_info);
    const file_name = std.fs.path.basename(path_info);
    var file_dir: std.fs.Dir = undefined;
    defer file_dir.close();
    if (dir_path != null) {
        file_dir = try std.fs.cwd().makeOpenPath(dir_path.?, .{});
    } else {
        file_dir = std.fs.cwd();
    }
    var file = try file_dir.openFile(file_name, .{});
    defer file.close();
    const file_size = (try file.stat()).size;
    const buf = try allocator.alloc(u8, file_size);
    defer allocator.free(buf);
    try file.reader().readNoEof(buf);
    var iterator = std.mem.splitScalar(u8, buf, '\n');
    const match: []const u8 = "Location: ";
    const path: ?[]const u8 = find_path: while (iterator.next()) |line| {
        if (std.mem.indexOf(u8, line, match)) |v| {
            break :find_path line[v + match.len ..];
        }
    } else null;

    if (path) |p| {
        const include: []const u8 = "/mlx/include";
        const lib: []const u8 = "/mlx/lib";
        var lib_path = try allocator.alloc(u8, p.len + lib.len);
        defer allocator.free(lib_path);
        @memcpy(lib_path[0..p.len], p);
        @memcpy(lib_path[p.len..], lib);
        c.addLibraryPath(.{ .path = lib_path });
        c.addRPath(.{ .path = lib_path });
        var include_path = try allocator.alloc(u8, p.len + include.len);
        defer allocator.free(include_path);
        @memcpy(include_path[0..p.len], p);
        @memcpy(include_path[p.len..], include);
        c.addIncludePath(.{ .path = include_path });
        c.linkSystemLibrary("mlx");
    } else {
        return error.MLXPathNotFound;
    }
}

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // install mlx via pip
    const install_mlx = b.addSystemCommand(&[_][]const u8{ "pip", "install", "mlx" });
    // get location of mlx install
    const mlx_loc = b.addSystemCommand(&[_][]const u8{ "pip", "show", "mlx" });
    mlx_loc.step.dependOn(&install_mlx.step);
    // write location of mlx to `mlx_info.txt` for use w include/linking
    b.getInstallStep().dependOn(&b.addInstallFileWithDir(mlx_loc.captureStdOut(), .prefix, "mlx_info.txt").step);
    const bindings_lib = b.addSharedLibrary(.{
        .name = "mlx_bindings",
        .root_source_file = .{ .path = "bindings/mlx.cc" },
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    bindings_lib.step.dependOn(&mlx_loc.step);
    bindings_lib.linkLibCpp();
    try addLibInfo(b.allocator, bindings_lib, "zig-out/mlx_info.txt");
    b.installArtifact(bindings_lib);

    const main_module = b.addModule("zigMLX", .{
        .source_file = .{ .path = "src/mlx.zig" },
    });

    const lib = b.addStaticLibrary(.{
        .name = "zigMLX",
        .root_source_file = .{ .path = "src/mlx.zig" },
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    lib.addModule("zigMLX", main_module);
    lib.step.dependOn(&bindings_lib.step);
    lib.addRPath(.{ .path = "zig-out/lib" });
    lib.addLibraryPath(.{ .path = "zig-out/lib" });
    lib.addIncludePath(.{ .path = "bindings" });
    lib.linkSystemLibrary("mlx_bindings");
    b.installArtifact(lib);

    // Unit Tests
    const main_tests = b.addTest(.{
        .root_source_file = .{ .path = "src/mlx.zig" },
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    main_tests.addModule("zigMLX", main_module);
    main_tests.step.dependOn(&bindings_lib.step);
    main_tests.addRPath(.{ .path = "zig-out/lib" });
    main_tests.addLibraryPath(.{ .path = "zig-out/lib" });
    main_tests.addIncludePath(.{ .path = "bindings" });
    main_tests.linkSystemLibrary("mlx_bindings");
    b.installArtifact(main_tests);

    const run_main_tests = b.addRunArtifact(main_tests);

    const test_step = b.step("test", "Run library tests");
    test_step.dependOn(&run_main_tests.step);

    const clang_fmt = b.addSystemCommand(&[_][]const u8{ "clang-format", "-i", "bindings/mlx_types.h", "bindings/mlx.cc", "bindings/mlx.h" });
    const zig_fmt = b.addSystemCommand(&[_][]const u8{ "zig", "fmt", "." });
    zig_fmt.step.dependOn(&clang_fmt.step);
    test_step.dependOn(&run_main_tests.step);

    const fmt_step = b.step("fmt", "Format library C and Zig code");
    fmt_step.dependOn(&zig_fmt.step);
}
