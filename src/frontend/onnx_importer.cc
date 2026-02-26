#include "onnx_importer.hpp"

#include <ranges>
#include <vector>

#include "types.hpp"

namespace tensor_compiler::frontend {

ir::Graph ImportOnnx(const onnx::GraphProto& onnx_graph) {
    ir::Graph graph;
    
    for (const auto& in : onnx_graph.input()) {
        graph.MarkGraphInput(graph.GetOrCreateValue(in.name()));
    }

    for (const auto& init : onnx_graph.initializer()) {
        graph.MarkInitializer(graph.GetOrCreateValue(init.name()));
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
