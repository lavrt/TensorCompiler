#pragma once

#include <onnx/onnx_pb.h>

#include "graph.hpp"

namespace tensor_compiler::frontend {

ir::Graph ImportOnnx(const onnx::GraphProto& model);

} // namespace tensor_compiler::frontend
