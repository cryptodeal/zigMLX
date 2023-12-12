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

mlx_err data(void **res, mlx_array arr) {
  std::exception_ptr eptr;
  try {
    auto a = static_cast<array *>(arr);
    void *dptr;
    switch (a->dtype()) {
    case mlx::core::bool_: {
      dptr = a->data<bool>();
      break;
    }
    case mlx::core::uint8: {
      dptr = a->data<uint8_t>();
      break;
    }
    case mlx::core::uint16: {
      dptr = a->data<uint16_t>();
      break;
    }
    case mlx::core::uint32: {
      dptr = a->data<uint32_t>();
      break;
    }
    case mlx::core::uint64: {
      dptr = a->data<uint64_t>();
      break;
    }
    case mlx::core::int8: {
      dptr = a->data<int8_t>();
      break;
    }
    case mlx::core::int16: {
      dptr = a->data<int16_t>();
      break;
    }
    case mlx::core::int32: {
      dptr = a->data<int32_t>();
      break;
    }
    case mlx::core::int64: {
      dptr = a->data<int64_t>();
      break;
    }
    case mlx::core::float16: {
      dptr = a->data<float16_t>();
      break;
    }
    case mlx::core::float32: {
      dptr = a->data<float>();
      break;
    }
    case mlx::core::bfloat16: {
      dptr = a->data<bfloat16_t>();
      break;
    }
    // TODO: case mlx_dtype::complex64:
    default: {
      throw std::invalid_argument("Unhandled dtype");
    }
    }
    std::swap(*res, dptr);
  } catch (...) {
    eptr = std::current_exception(); // capture
  }
  return handle_eptr(eptr);
}

mlx_err fromPtr(mlx_array *res, const void *data, const int *shape, size_t shape_len,
                mlx_dtype dtype) {
  std::exception_ptr eptr;
  try {
    std::vector<int> shape_vec(shape, shape + shape_len);
    mlx_array new_array;
    switch (dtype) {
    case mlx_dtype::bool_: {
      auto arr = array(static_cast<const bool *>(data), shape_vec, mlx::core::bool_);
      new_array = new array(arr);
      break;
    }
    case mlx_dtype::uint8: {
      auto arr =
          array(static_cast<const uint8_t *>(data), shape_vec, mlx::core::uint8);
      new_array = new array(arr);
      break;
    }
    case mlx_dtype::uint16: {
      auto arr =
          array(static_cast<const uint16_t *>(data), shape_vec, mlx::core::uint16);
      new_array = new array(arr);
      break;
    }
    case mlx_dtype::uint32: {
      auto arr =
          array(static_cast<const uint32_t *>(data), shape_vec, mlx::core::uint32);
      new_array = new array(arr);
      break;
    }
    case mlx_dtype::uint64: {
      auto arr =
          array(static_cast<const uint64_t *>(data), shape_vec, mlx::core::uint64);
      new_array = new array(arr);
      break;
    }
    case mlx_dtype::int8: {
      auto arr = array(static_cast<const int8_t *>(data), shape_vec, mlx::core::int8);
      new_array = new array(arr);
      break;
    }
    case mlx_dtype::int16: {
      auto arr =
          array(static_cast<const int16_t *>(data), shape_vec, mlx::core::int16);
      new_array = new array(arr);
      break;
    }
    case mlx_dtype::int32: {
      auto arr =
          array(static_cast<const int32_t *>(data), shape_vec, mlx::core::int32);
      new_array = new array(arr);
      break;
    }
    case mlx_dtype::int64: {
      auto arr =
          array(static_cast<const int64_t *>(data), shape_vec, mlx::core::int64);
      new_array = new array(arr);
      break;
    }
    case mlx_dtype::float16: {
      auto arr =
          array(static_cast<const float16_t *>(data), shape_vec, mlx::core::float16);
      new_array = new array(arr);
      break;
    }
    case mlx_dtype::float32: {
      auto arr =
          array(static_cast<const float *>(data), shape_vec, mlx::core::float32);
      new_array = new array(arr);
      break;
    }
    case mlx_dtype::bfloat16: {
      auto arr = array(static_cast<const bfloat16_t *>(data), shape_vec,
                       mlx::core::bfloat16);
      new_array = new array(arr);
      break;
    }
    // TODO: case mlx_dtype::complex64:
    default: {
      throw std::invalid_argument("Invalid dtype enum: " +
                                  std::to_string(dtype));
    }
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
