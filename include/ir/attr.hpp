#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace tensor_compiler::ir {

using AttrValue = std::variant<
    std::int64_t,
    double,
    std::string,
    std::vector<std::int64_t>,
    std::vector<double>
>;

using AttrMap = std::unordered_map<std::string, AttrValue>;

} // namespace tensor_compiler::ir
