#include <cmath>
#include <exception>
#include <iostream>

#include "mlx/mlx.h"
#include "mlx_types.h"

using namespace mlx::core;

mlx_err handle_eptr(std::exception_ptr eptr) {
  try {
    if (eptr)
      std::rethrow_exception(eptr);
  } catch (const std::exception &e) {
    std::cout << "Caught exception: '" << e.what() << "'\n";
    return mlx_err::mlx_exception;
  }
  return mlx_err::mlx_success;
}

Dtype dtypeFromEnum(mlx_dtype dtype_enum) {
  switch (dtype_enum) {
  case mlx_dtype::bool_:
    return mlx::core::bool_;
  case mlx_dtype::uint8:
    return mlx::core::uint8;
  case mlx_dtype::uint16:
    return mlx::core::uint16;
  case mlx_dtype::uint32:
    return mlx::core::uint32;
  case mlx_dtype::uint64:
    return mlx::core::uint64;
  case mlx_dtype::int8:
    return mlx::core::int8;
  case mlx_dtype::int16:
    return mlx::core::int16;
  case mlx_dtype::int32:
    return mlx::core::int32;
  case mlx_dtype::int64:
    return mlx::core::int64;
  case mlx_dtype::float16:
    return mlx::core::float16;
  case mlx_dtype::float32:
    return mlx::core::float32;
  case mlx_dtype::bfloat16:
    return mlx::core::bfloat16;
  // TODO: case mlx_dtype::complex64:
  default:
    throw std::invalid_argument("Invalid dtype enum");
  }
}

extern "C" {

void destroyArray(mlx_array array_handle) {
  delete static_cast<array *>(array_handle);
}

mlx_err seed(uint64_t seed) {
  std::exception_ptr eptr;
  try {
    random::seed(seed);
  } catch (...) {
    eptr = std::current_exception(); // capture
  }
  return handle_eptr(eptr);
}

mlx_err fromScalar(mlx_array *res, double val, mlx_dtype dtype) {
  std::exception_ptr eptr;
  try {
    auto arr = array(val, dtypeFromEnum(dtype));
    mlx_array new_array = new array(arr);
    std::swap(*res, new_array);
  } catch (...) {
    eptr = std::current_exception(); // capture
  }
  return handle_eptr(eptr);
}

mlx_err fromScalarI64(mlx_array *res, int64_t val) {
  std::exception_ptr eptr;
  try {
    auto arr = array(val);
    mlx_array new_array = new array(arr);
    std::swap(*res, new_array);
  } catch (...) {
    eptr = std::current_exception(); // capture
  }
  return handle_eptr(eptr);
}

mlx_err fromScalarU64(mlx_array *res, uint64_t val) {
  std::exception_ptr eptr;
  try {
    auto arr = array(val);
    mlx_array new_array = new array(arr);
    std::swap(*res, new_array);
  } catch (...) {
    eptr = std::current_exception(); // capture
  }
  return handle_eptr(eptr);
}

mlx_err fromPtr(mlx_array *res, void *data, const int *shape, size_t shape_len,
                mlx_dtype dtype) {
  std::exception_ptr eptr;
  try {
    std::vector<int> shape_vec(shape, shape + shape_len);
    mlx_array new_array;
    switch (dtype) {
    case mlx_dtype::bool_: {
      auto arr =
          array(static_cast<bool *>(data), shape_vec, dtypeFromEnum(dtype));
      new_array = new array(arr);
    }
    case mlx_dtype::uint8: {
      auto arr =
          array(static_cast<uint8_t *>(data), shape_vec, dtypeFromEnum(dtype));
      new_array = new array(arr);
    }
    case mlx_dtype::uint16: {
      auto arr =
          array(static_cast<uint16_t *>(data), shape_vec, dtypeFromEnum(dtype));
      new_array = new array(arr);
    }
    case mlx_dtype::uint32: {
      auto arr =
          array(static_cast<uint32_t *>(data), shape_vec, dtypeFromEnum(dtype));
      new_array = new array(arr);
    }
    case mlx_dtype::uint64: {
      auto arr =
          array(static_cast<uint64_t *>(data), shape_vec, dtypeFromEnum(dtype));
      new_array = new array(arr);
    }
    case mlx_dtype::int8: {
      auto arr =
          array(static_cast<int8_t *>(data), shape_vec, dtypeFromEnum(dtype));
      new_array = new array(arr);
    }
    case mlx_dtype::int16: {
      auto arr =
          array(static_cast<int16_t *>(data), shape_vec, dtypeFromEnum(dtype));
      new_array = new array(arr);
    }
    case mlx_dtype::int32: {
      auto arr =
          array(static_cast<int32_t *>(data), shape_vec, dtypeFromEnum(dtype));
      new_array = new array(arr);
    }
    case mlx_dtype::int64: {
      auto arr =
          array(static_cast<int64_t *>(data), shape_vec, dtypeFromEnum(dtype));
      new_array = new array(arr);
    }
    case mlx_dtype::float16: {
      auto arr = array(static_cast<float16_t *>(data), shape_vec,
                       dtypeFromEnum(dtype));
      new_array = new array(arr);
    }
    case mlx_dtype::float32: {
      auto arr =
          array(static_cast<float *>(data), shape_vec, dtypeFromEnum(dtype));
      new_array = new array(arr);
    }
    case mlx_dtype::bfloat16: {
      auto arr = array(static_cast<bfloat16_t *>(data), shape_vec,
                       dtypeFromEnum(dtype));
      new_array = new array(arr);
    }
    // TODO: case mlx_dtype::complex64:
    default:
      throw std::invalid_argument("Invalid dtype enum");
    }
    std::swap(*res, new_array);
  } catch (...) {
    eptr = std::current_exception(); // capture
  }
  return handle_eptr(eptr);
}

mlx_err randomNormal(mlx_array *res, const int *shape, size_t shape_len,
                     mlx_dtype dtype) {
  std::exception_ptr eptr;

  try {
    std::vector<int> shape_vec(shape, shape + shape_len);
    auto arr = random::normal(shape_vec, dtypeFromEnum(dtype));
    mlx_array new_array = new array(arr);
    std::swap(*res, new_array);
  } catch (...) {
    eptr = std::current_exception(); // capture
  }
  return handle_eptr(eptr);
}
}
