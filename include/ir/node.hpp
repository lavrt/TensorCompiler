#pragma once

#include <vector>
#include <string>

#include "attr.hpp"
#include "types.hpp"

namespace tensor_compiler::ir {

enum class OpCode {
    kUnknown = 0,
    kAdd = 1,
    kMul = 2,
    kConv = 3,
    kRelu = 4,
    kMatMul = 5,
    kGemm = 6,
};

inline OpCode ParseOpCode(std::string_view op) noexcept {
    if (op == "Add") return OpCode::kAdd;
    if (op == "Mul") return OpCode::kMul;
    if (op == "Conv") return OpCode::kConv;
    if (op == "Relu") return OpCode::kRelu;
    if (op == "MatMul") return OpCode::kMatMul;
    if (op == "Gemm") return OpCode::kGemm;
    return OpCode::kUnknown;
}

struct Node final {
    std::string name;

    std::string op_name;
    OpCode op;

    std::vector<ValueId> inputs;
    std::vector<ValueId> outputs;

    AttrMap attrs;

    Node(std::string name_,
         std::string op_name_,
         std::vector<ValueId>&& in,
         std::vector<ValueId>&& out
         )
        : name(std::move(name_)),
          op_name(std::move(op_name_)),
          op(ParseOpCode(op_name)),
          inputs(std::move(in)),
          outputs(std::move(out))
    {}
};

} // namespace tensor_compiler::ir
