#pragma once

#include <string>

#include <onnx/onnx_pb.h>

namespace tensor_compiler {

onnx::ModelProto LoadOnnxModel(const std::string& path);

} // namespace tensor_compiler