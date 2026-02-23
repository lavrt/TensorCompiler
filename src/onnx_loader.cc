#include "onnx_loader.hpp"

#include <fstream>
#include <stdexcept>

#include <google/protobuf/io/zero_copy_stream_impl.h>

namespace tensor_compiler {

onnx::ModelProto LoadOnnxModel(const std::string& path) {
    std::ifstream in{path, std::ios::binary};
    if (!in) {
        throw std::runtime_error("Cannot open file: " + path);
    }

    google::protobuf::io::IstreamInputStream zcis{&in};
    onnx::ModelProto model;
    if (!model.ParseFromZeroCopyStream(&zcis)) {
        throw std::runtime_error("Failed to parse ONNX");
    }

    return model;
}

} // namespace tensor_compiler
