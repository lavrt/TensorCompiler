#pragma once

#include <cstdint>
#include <vector>

namespace tensor_compiler::ir {

enum class DType {
    kUnknown = 0,
    kF32,
    kI8, kI32, kI64,
    kU8,
};

struct TensorData {
    DType dtype = DType::kUnknown;
    std::vector<std::int64_t> shape;
};

} // namespace tensor_compiler::ir
