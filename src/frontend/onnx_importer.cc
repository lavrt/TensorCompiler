#include "onnx_importer.hpp"

#include <ranges>
#include <vector>

#include "tensor.hpp"
#include "types.hpp"

namespace {

tensor_compiler::ir::DType ToIrDType(int dt) {
    namespace tc = tensor_compiler;
    switch (dt) {
        case onnx::TensorProto_DataType_FLOAT: return tc::ir::DType::kF32;
        case onnx::TensorProto_DataType_INT64: return tc::ir::DType::kI64;
        case onnx::TensorProto_DataType_INT32: return tc::ir::DType::kI32;
        case onnx::TensorProto_DataType_INT8:  return tc::ir::DType::kI8;
        case onnx::TensorProto_DataType_UINT8: return tc::ir::DType::kU8;
        default: return tc::ir::DType::kUnknown;
    }
}

} // namespace

namespace tensor_compiler::frontend {

ir::Graph ImportOnnx(const onnx::GraphProto& onnx_graph) {
    ir::Graph graph;
    
    for (const auto& in : onnx_graph.input()) {
        graph.MarkGraphInput(graph.GetOrCreateValue(in.name()));
    }

    for (const auto& init : onnx_graph.initializer()) {
        auto vid = graph.GetOrCreateValue(init.name());
        graph.MarkInitializer(vid);
        ir::TensorData t{
            .dtype = ToIrDType(init.data_type()),
            .shape = {init.dims().begin(), init.dims().end()}
        };
        graph.constants.insert({vid, std::move(t)});
    }

    for (const auto& out : onnx_graph.output()) {
        graph.MarkGraphOutput(graph.GetOrCreateValue(out.name()));
    }

    for (const auto& onnx_node : onnx_graph.node()) {
        graph.AddNode({
            onnx_node.name(), onnx_node.op_type(),

            onnx_node.input() | std::views::transform([&graph](const auto& name) {
                return graph.GetOrCreateValue(name);
            }) | std::ranges::to<std::vector<ir::ValueId>>(),

            onnx_node.output() | std::views::transform([&graph](const auto& name) {
                return graph.GetOrCreateValue(name);
            }) | std::ranges::to<std::vector<ir::ValueId>>()
        });
    }

    return graph;
}

} // namespace tensor_compiler::frontend
