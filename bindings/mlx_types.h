#pragma once

typedef enum {
  mlx_success,
  mlx_exception,
} mlx_err;

typedef enum {
  bool_,
  uint8,
  uint16,
  uint32,
  uint64,
  int8,
  int16,
  int32,
  int64,
  float16,
  float32,
  bfloat16,
  complex64,
} mlx_dtype;

typedef struct mlx_array_flags {
  bool contiguous;

  bool row_contiguous;
  bool col_contiguous;
} mlx_array_flags;

typedef void *mlx_array;
typedef void *mlx_array_iterator;
typedef void *mlx_primitive;
