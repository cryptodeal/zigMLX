#include <cmath>
#include <exception>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdlib.h>

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

mlx_dtype enumFromDtype(Dtype val) {
  switch (val) {
  case mlx::core::bool_:
    return mlx_dtype::bool_;
  case mlx::core::uint8:
    return mlx_dtype::uint8;
  case mlx::core::uint16:
    return mlx_dtype::uint16;
  case mlx::core::uint32:
    return mlx_dtype::uint32;
  case mlx::core::uint64:
    return mlx_dtype::uint64;
  case mlx::core::int8:
    return mlx_dtype::int8;
  case mlx::core::int16:
    return mlx_dtype::int16;
  case mlx::core::int32:
    return mlx_dtype::int32;
  case mlx::core::int64:
    return mlx_dtype::int64;
  case mlx::core::float16:
    return mlx_dtype::float16;
  case mlx::core::float32:
    return mlx_dtype::float32;
  case mlx::core::bfloat16:
    return mlx_dtype::bfloat16;
  // TODO: case mlx_dtype::complex64:
  default:
    throw std::invalid_argument("Invalid dtype enum");
  }
}

extern "C" {

void destroyArray(mlx_array arr) { delete static_cast<array *>(arr); }

void destroyArrayIterator(mlx_array_iterator iter) {
  delete static_cast<array::ArrayIterator*>(iter);
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

mlx_err nextDiff(mlx_array_iterator iter, size_t diff) {
  std::exception_ptr eptr;
  try {
    auto i = static_cast<array::ArrayIterator *>(iter);
    auto tmp = i + diff;
  } catch (...) {
    eptr = std::current_exception(); // capture
  }
  return handle_eptr(eptr);
}

mlx_err next(mlx_array_iterator iter) {
  std::exception_ptr eptr;
  try {
    auto i = static_cast<array::ArrayIterator *>(iter);
    i++;
  } catch (...) {
    eptr = std::current_exception(); // capture
  }
  return handle_eptr(eptr);
}

mlx_err arrayIterEql(bool *res, mlx_array_iterator a, mlx_array_iterator b) {
  std::exception_ptr eptr;
  try {
    auto iter_a = static_cast<array::ArrayIterator *>(a);
    auto iter_b = static_cast<array::ArrayIterator *>(b);
    *res = (*iter_a) == (*iter_b);
  } catch (...) {
    eptr = std::current_exception(); // capture
  }
  return handle_eptr(eptr);
}

mlx_err arrayIterNeq(bool *res, mlx_array_iterator a, mlx_array_iterator b) {
  std::exception_ptr eptr;
  try {
    auto iter_a = static_cast<array::ArrayIterator *>(a);
    auto iter_b = static_cast<array::ArrayIterator *>(b);
    *res = (*iter_a) != (*iter_b);
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
    mlx_array new_array = new array(val, mlx::core::int64);
    std::swap(*res, new_array);
  } catch (...) {
    eptr = std::current_exception(); // capture
  }
  return handle_eptr(eptr);
}

mlx_err fromScalarU64(mlx_array *res, uint64_t val) {
  std::exception_ptr eptr;
  try {
    mlx_array new_array = new array(val, mlx::core::uint64);
    std::swap(*res, new_array);
  } catch (...) {
    eptr = std::current_exception(); // capture
  }
  return handle_eptr(eptr);
}

mlx_err initHandle(mlx_array *res, const void *shape, size_t shape_len,
                   mlx_dtype dtype) {
  std::exception_ptr eptr;
  try {
    auto shape_int = reinterpret_cast<const int *>(shape);
    std::vector<int> shape_vec(shape_int, shape_int + shape_len);
    auto data_type = dtypeFromEnum(dtype);
    const int product = std::accumulate(shape_vec.begin(), shape_vec.end(), 1,
                                        std::multiplies<int>());
    auto buffer = allocator::malloc(product * size_of(data_type));
    mlx_array new_array = new array(buffer, shape_vec, data_type);
    std::swap(*res, new_array);
  } catch (...) {
    eptr = std::current_exception(); // capture
  }
  return handle_eptr(eptr);
}

mlx_err initEmpty(mlx_array *res) {
  std::exception_ptr eptr;
  try {
    mlx_array new_array = new array({});
    std::swap(*res, new_array);
  } catch (...) {
    eptr = std::current_exception(); // capture
  }
  return handle_eptr(eptr);
}

mlx_err fromPtr(mlx_array *res, const void *data, const void *shape,
                size_t shape_len, mlx_dtype dtype) {
  std::exception_ptr eptr;
  try {
    auto shape_int = reinterpret_cast<const int *>(shape);
    std::vector<int> shape_vec(shape_int, shape_int + shape_len);
    mlx_array new_array;
    switch (dtype) {
    case mlx_dtype::bool_: {
      new_array = new array(static_cast<const bool *>(data), shape_vec,
                            mlx::core::bool_);
      break;
    }
    case mlx_dtype::uint8: {
      new_array = new array(static_cast<const uint8_t *>(data), shape_vec,
                            mlx::core::uint8);
      break;
    }
    case mlx_dtype::uint16: {
      new_array = new array(static_cast<const uint16_t *>(data), shape_vec,
                            mlx::core::uint16);
      break;
    }
    case mlx_dtype::uint32: {
      new_array = new array(static_cast<const uint32_t *>(data), shape_vec,
                            mlx::core::uint32);
      break;
    }
    case mlx_dtype::uint64: {
      new_array = new array(static_cast<const uint64_t *>(data), shape_vec,
                            mlx::core::uint64);
      break;
    }
    case mlx_dtype::int8: {
      new_array = new array(static_cast<const int8_t *>(data), shape_vec,
                            mlx::core::int8);
      break;
    }
    case mlx_dtype::int16: {
      new_array = new array(static_cast<const int16_t *>(data), shape_vec,
                            mlx::core::int16);
      break;
    }
    case mlx_dtype::int32: {
      new_array = new array(static_cast<const int32_t *>(data), shape_vec,
                            mlx::core::int32);
      break;
    }
    case mlx_dtype::int64: {
      new_array = new array(static_cast<const int64_t *>(data), shape_vec,
                            mlx::core::int64);
      break;
    }
    case mlx_dtype::float16: {
      new_array = new array(static_cast<const float16_t *>(data), shape_vec,
                            mlx::core::float16);
      break;
    }
    case mlx_dtype::float32: {
      new_array = new array(static_cast<const float *>(data), shape_vec,
                            mlx::core::float32);
      break;
    }
    case mlx_dtype::bfloat16: {
      new_array = new array(static_cast<const bfloat16_t *>(data), shape_vec,
                            mlx::core::bfloat16);
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

mlx_err randomNormal(mlx_array *res, const void *shape, size_t shape_len,
                     mlx_dtype dtype) {
  std::exception_ptr eptr;

  try {
    auto shape_int = reinterpret_cast<const int *>(shape);
    std::vector<int> shape_vec(shape_int, shape_int + shape_len);
    auto arr = random::normal(shape_vec, dtypeFromEnum(dtype));
    mlx_array new_array = new array(arr);
    std::swap(*res, new_array);
  } catch (...) {
    eptr = std::current_exception(); // capture
  }
  return handle_eptr(eptr);
}

mlx_err itemsize(size_t *res, mlx_array arr) {
  std::exception_ptr eptr;
  try {
    auto a = static_cast<array *>(arr);
    *res = a->itemsize();
  } catch (...) {
    eptr = std::current_exception(); // capture
  }
  return handle_eptr(eptr);
}

mlx_err size(size_t *res, mlx_array arr) {
  std::exception_ptr eptr;
  try {
    auto a = static_cast<array *>(arr);
    *res = a->size();
  } catch (...) {
    eptr = std::current_exception(); // capture
  }
  return handle_eptr(eptr);
}

mlx_err nbytes(size_t *res, mlx_array arr) {
  std::exception_ptr eptr;
  try {
    auto a = static_cast<array *>(arr);
    *res = a->nbytes();
  } catch (...) {
    eptr = std::current_exception(); // capture
  }
  return handle_eptr(eptr);
}

mlx_err ndim(size_t *res, mlx_array arr) {
  std::exception_ptr eptr;
  try {
    auto a = static_cast<array *>(arr);
    *res = a->ndim();
  } catch (...) {
    eptr = std::current_exception(); // capture
  }
  return handle_eptr(eptr);
}

mlx_err shape(void **res, mlx_array arr) {
  std::exception_ptr eptr;
  try {
    auto a = static_cast<array *>(arr);
    std::vector<int> a_shape = a->shape();
    *res = a_shape.data();
  } catch (...) {
    eptr = std::current_exception(); // capture
  }
  return handle_eptr(eptr);
}

mlx_err dim(int *res, int dimension, mlx_array arr) {
  std::exception_ptr eptr;
  try {
    auto a = static_cast<array *>(arr);
    *res = a->shape(dimension);
  } catch (...) {
    eptr = std::current_exception(); // capture
  }
  return handle_eptr(eptr);
}

mlx_err strides(void **res, size_t *stride_len, mlx_array arr) {
  std::exception_ptr eptr;
  try {
    auto a = static_cast<array *>(arr);
    std::vector<size_t> a_strides = a->strides();
    *stride_len = a_strides.size();
    *res = a_strides.data();
  } catch (...) {
    eptr = std::current_exception(); // capture
  }
  return handle_eptr(eptr);
}

mlx_err dtype(mlx_dtype *res, mlx_array arr) {
  std::exception_ptr eptr;
  try {
    auto a = static_cast<array *>(arr);
    *res = enumFromDtype(a->dtype());
  } catch (...) {
    eptr = std::current_exception(); // capture
  }
  return handle_eptr(eptr);
}

mlx_err eval_array(bool retain_graph, mlx_array arr) {
  std::exception_ptr eptr;
  try {
    auto a = static_cast<array *>(arr);
    a->eval(retain_graph);
  } catch (...) {
    eptr = std::current_exception(); // capture
  }
  return handle_eptr(eptr);
}

mlx_err item(void *res, bool retain_graph, mlx_array arr) {
  std::exception_ptr eptr;
  try {
    auto a = static_cast<array *>(arr);
    switch (a->dtype()) {
    case mlx::core::bool_: {
      *(static_cast<bool *>(res)) = a->item<bool>(retain_graph);
      break;
    }
    case mlx::core::uint8: {
      *(static_cast<uint8_t *>(res)) = a->item<uint8_t>(retain_graph);
      break;
    }
    case mlx::core::uint16: {
      *(static_cast<uint16_t *>(res)) = a->item<uint16_t>(retain_graph);
      break;
    }
    case mlx::core::uint32: {
      *(static_cast<uint32_t *>(res)) = a->item<uint32_t>(retain_graph);
      break;
    }
    case mlx::core::uint64: {
      *(static_cast<uint64_t *>(res)) = a->item<uint64_t>(retain_graph);
      break;
    }
    case mlx::core::int8: {
      *(static_cast<int8_t *>(res)) = a->item<int8_t>(retain_graph);
      break;
    }
    case mlx::core::int16: {
      *(static_cast<int16_t *>(res)) = a->item<int16_t>(retain_graph);
      break;
    }
    case mlx::core::int32: {
      *(static_cast<int32_t *>(res)) = a->item<int32_t>(retain_graph);
      break;
    }
    case mlx::core::int64: {
      *(static_cast<int64_t *>(res)) = a->item<int64_t>(retain_graph);
      break;
    }
    case mlx::core::float16: {
      *(static_cast<float16_t *>(res)) = a->item<float16_t>(retain_graph);
      break;
    }
    case mlx::core::float32: {
      *(static_cast<float *>(res)) = a->item<float>(retain_graph);
      break;
    }
    case mlx::core::bfloat16: {
      *(static_cast<bfloat16_t *>(res)) = a->item<bfloat16_t>(retain_graph);
      break;
    }
    // TODO: case mlx_dtype::complex64:
    default: {
      throw std::invalid_argument("Unhandled dtype");
    }
    }
  } catch (...) {
    eptr = std::current_exception(); // capture
  }
  return handle_eptr(eptr);
}

mlx_err begin(mlx_array_iterator *res, mlx_array arr) {
  std::exception_ptr eptr;
  try {
    auto a = static_cast<array *>(arr);
    array::ArrayIterator *iter = new array::ArrayIterator(*a);
    *res = iter;
  } catch (...) {
    eptr = std::current_exception(); // capture
  }
  return handle_eptr(eptr);
}

mlx_err end(mlx_array_iterator *res, mlx_array arr) {
  std::exception_ptr eptr;
  try {
    auto a = static_cast<array *>(arr);
    array::ArrayIterator *iter = new array::ArrayIterator(*a, a->shape(0));
    *res = iter;
  } catch (...) {
    eptr = std::current_exception(); // capture
  }
  return handle_eptr(eptr);
}

mlx_err id(size_t *res, mlx_array arr) {
  std::exception_ptr eptr;
  try {
    auto a = static_cast<array *>(arr);
    *res = a->id();
  } catch (...) {
    eptr = std::current_exception(); // capture
  }
  return handle_eptr(eptr);
}

mlx_err primitive(mlx_primitive *res, mlx_array arr) {
  std::exception_ptr eptr;
  try {
    auto a = static_cast<array *>(arr);
    *res = &(a->primitive());
  } catch (...) {
    eptr = std::current_exception(); // capture
  }
  return handle_eptr(eptr);
}

mlx_err has_primitive(bool *res, mlx_array arr) {
  std::exception_ptr eptr;
  try {
    auto a = static_cast<array *>(arr);
    *res = a->has_primitive();
  } catch (...) {
    eptr = std::current_exception(); // capture
  }
  return handle_eptr(eptr);
}

// TODO: mlx_err inputs() {}

// TODO: mlx_err editable_inputs() {}

mlx_err detach(mlx_array arr) {
  std::exception_ptr eptr;
  try {
    auto a = static_cast<array *>(arr);
    a->detach();
  } catch (...) {
    eptr = std::current_exception(); // capture
  }
  return handle_eptr(eptr);
}

mlx_err flags(mlx_array_flags *res, mlx_array arr) {
  std::exception_ptr eptr;
  try {
    auto a = static_cast<array *>(arr);
    auto flags = a->flags();
    res->contiguous = flags.contiguous;
    res->row_contiguous = flags.row_contiguous;
    res->col_contiguous = flags.col_contiguous;
  } catch (...) {
    eptr = std::current_exception(); // capture
  }
  return handle_eptr(eptr);
}

mlx_err data_size(size_t *res, mlx_array arr) {
  std::exception_ptr eptr;
  try {
    auto a = static_cast<array *>(arr);
    *res = a->data_size();
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

mlx_err is_evaled(bool *res, mlx_array arr) {
  std::exception_ptr eptr;
  try {
    auto a = static_cast<array *>(arr);
    *res = a->is_evaled();
  } catch (...) {
    eptr = std::current_exception(); // capture
  }
  return handle_eptr(eptr);
}

mlx_err set_tracer(bool is_tracer, mlx_array arr) {
  std::exception_ptr eptr;
  try {
    auto a = static_cast<array *>(arr);
    a->set_tracer(is_tracer);
  } catch (...) {
    eptr = std::current_exception(); // capture
  }
  return handle_eptr(eptr);
}

mlx_err add(mlx_array *res, mlx_array lhs, mlx_array rhs) {
  std::exception_ptr eptr;
  try {
    auto lhs_array = static_cast<array *>(lhs);
    auto rhs_array = static_cast<array *>(rhs);
    array tmp = mlx::core::add(*lhs_array, *rhs_array);
    *res = new array(tmp);
  } catch (...) {
    eptr = std::current_exception(); // capture
  }
  return handle_eptr(eptr);
}

mlx_err subtract(mlx_array *res, mlx_array lhs, mlx_array rhs) {
  std::exception_ptr eptr;
  try {
    auto lhs_array = static_cast<array *>(lhs);
    auto rhs_array = static_cast<array *>(rhs);
    auto tmp = mlx::core::subtract(*lhs_array, *rhs_array);
    mlx_array new_array = new array(tmp);
    std::swap(*res, new_array);
  } catch (...) {
    eptr = std::current_exception(); // capture
  }
  return handle_eptr(eptr);
}

mlx_err multiply(mlx_array *res, mlx_array lhs, mlx_array rhs) {
  std::exception_ptr eptr;
  try {
    auto lhs_array = static_cast<array *>(lhs);
    auto rhs_array = static_cast<array *>(rhs);
    auto tmp = mlx::core::multiply(*lhs_array, *rhs_array);
    mlx_array new_array = new array(tmp);
    std::swap(*res, new_array);
  } catch (...) {
    eptr = std::current_exception(); // capture
  }
  return handle_eptr(eptr);
}

mlx_err divide(mlx_array *res, mlx_array lhs, mlx_array rhs) {
  std::exception_ptr eptr;
  try {
    auto lhs_array = static_cast<array *>(lhs);
    auto rhs_array = static_cast<array *>(rhs);
    auto tmp = mlx::core::divide(*lhs_array, *rhs_array);
    mlx_array new_array = new array(tmp);
    std::swap(*res, new_array);
  } catch (...) {
    eptr = std::current_exception(); // capture
  }
  return handle_eptr(eptr);
}
}
