#pragma once
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "mlx_types.h"

// Methods to free underlying memory
void destroyArray(mlx_array arr);
void destroyArrayIterator(mlx_array_iterator iter);

// Seed the random number generator.
mlx_err seed(uint64_t seed);

// array::ArrayIterator methods
mlx_err nextDiff(mlx_array_iterator iter, size_t diff);
mlx_err next(mlx_array_iterator iter);
mlx_err arrayIterEql(bool *res, mlx_array_iterator a, mlx_array_iterator b);
mlx_err arrayIterNeq(bool *res, mlx_array_iterator a, mlx_array_iterator b);

// Methods to initialize new array
mlx_err fromScalar(mlx_array *res, double val, mlx_dtype dtype);
mlx_err fromScalarI64(mlx_array *res, int64_t val);
mlx_err fromScalarU64(mlx_array *res, uint64_t val);
mlx_err initHandle(mlx_array *res, const void *shape, size_t shape_len,
                   mlx_dtype dtype);
mlx_err initEmpty(mlx_array *res);
mlx_err fromPtr(mlx_array *res, const void *data, const void *shape,
                size_t shape_len, mlx_dtype dtype);
mlx_err randomNormal(mlx_array *res, const void *shape, size_t shape_len,
                     mlx_dtype dtype);

// Methods to get array properties
mlx_err itemsize(size_t *res, mlx_array arr);
mlx_err size(size_t *res, mlx_array arr);
mlx_err nbytes(size_t *res, mlx_array arr);
mlx_err ndim(size_t *res, mlx_array arr);
mlx_err shape(void **res, mlx_array arr);
mlx_err dim(int *res, int dimension, mlx_array arr);
mlx_err strides(void **res, size_t *stride_len, mlx_array arr);
mlx_err dtype(mlx_dtype *res, mlx_array arr);

// Other array methods
mlx_err eval_array(bool retain_graph, mlx_array arr);
mlx_err item(void *res, bool retain_graph, mlx_array arr);
mlx_err begin(mlx_array_iterator *res, mlx_array arr);
mlx_err end(mlx_array_iterator *res, mlx_array arr);
mlx_err id(size_t *res, mlx_array arr);
mlx_err primitive(mlx_primitive *res, mlx_array arr);
mlx_err has_primitive(bool *res, mlx_array arr);
// TODO: mlx_err inputs();
// TODO: mlx_err editable_inputs();
mlx_err detach(mlx_array arr);
mlx_err flags(mlx_array_flags *res, mlx_array arr);
mlx_err data_size(size_t *res, mlx_array arr);
mlx_err data(void **res, mlx_array arr);
mlx_err is_evaled(bool *res, mlx_array arr);
mlx_err set_tracer(bool is_tracer, mlx_array arr);

// Ops on arrays
mlx_err add(mlx_array *res, mlx_array a, mlx_array b);
mlx_err subtract(mlx_array *res, mlx_array a, mlx_array b);
mlx_err multiply(mlx_array *res, mlx_array a, mlx_array b);
mlx_err divide(mlx_array *res, mlx_array a, mlx_array b);