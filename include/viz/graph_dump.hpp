#pragma once

#include <string>

#include "graph.hpp"

namespace tensor_compiler::viz {

std::string GraphDump(const ir::Graph& g);

} // namespace tensor_compiler::viz
