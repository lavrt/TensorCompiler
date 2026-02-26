#pragma once

#include <optional>
#include <string>
#include <vector>

#include "types.hpp"

namespace tensor_compiler::ir {

struct Value final {
    std::string name;

    bool is_graph_input = false;
    bool is_graph_output = false;
    bool is_initializer = false;
    
    std::optional<NodeId> producer;
    std::vector<NodeId> consumers;

    Value(std::string name_ = {}) : name(std::move(name_)) {}
};

} // namespace tensor_compiler::ir
