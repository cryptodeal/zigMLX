#pragma once
#include <stddef.h>
#include <stdint.h>

#include "mlx_types.h"

void destroyArray(void *array_handle);
mlx_err seed(uint64_t seed);
mlx_err fromScalar(mlx_array *res, double val, mlx_dtype dtype);
mlx_err fromScalarI64(mlx_array *res, int64_t val);
mlx_err fromScalarU64(mlx_array *res, uint64_t val);
mlx_err fromPtr(mlx_array *res, void *data, const int *shape, size_t shape_len,
                mlx_dtype dtype);
mlx_err randomNormal(mlx_array *res, const int *shape, size_t shape_len,
                     mlx_dtype dtype);