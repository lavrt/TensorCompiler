#include "onnx_loader.hpp"

#include <fstream>
#include <stdexcept>

#include <google/protobuf/io/zero_copy_stream_impl.h>

namespace tensor_compiler::frontend {

onnx::ModelProto LoadOnnxModel(const std::string& path) {
    std::ifstream fin{path, std::ios::binary};
    if (!fin) {
        throw std::runtime_error("Cannot open file: " + path);
    }

    google::protobuf::io::IstreamInputStream is{&fin};
    onnx::ModelProto model;
    if (!model.ParseFromZeroCopyStream(&is)) {
        throw std::runtime_error("Failed to parse ONNX");
    }

    return model;
}

} // namespace tensor_compiler::frontend
