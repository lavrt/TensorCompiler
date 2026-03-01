#include "onnx_importer.hpp"

#include <ranges>
#include <vector>

#include "attr.hpp"
#include "tensor.hpp"
#include "types.hpp"

namespace {

tensor_compiler::ir::AttrValue ParseAttr(const onnx::AttributeProto& a) {
    using A = onnx::AttributeProto;
    switch (a.type()) {
        case A::INT: {
            return static_cast<std::int64_t>(a.i());
        }
        case A::FLOAT: {
            return static_cast<double>(a.f());
        }
        case A::STRING: {
            return std::string(a.s());
        }
        case A::INTS: {
            std::vector<std::int64_t> v;
            v.reserve(a.ints_size());
            for (auto x : a.ints())
                v.push_back(static_cast<std::int64_t>(x));
            return v;
        }
        case A::FLOATS: {
            std::vector<double> v;
            v.reserve(a.floats_size());
            for (auto x : a.floats())
                v.push_back(static_cast<double>(x));
            return v;
        }
        default: {
            throw std::runtime_error("Unsupported ONNX attribute type for '" + a.name() + "'");
        }
    }
}

tensor_compiler::ir::AttrMap ParseAttrs(const onnx::NodeProto& n) {
    namespace tc = tensor_compiler;
    tc::ir::AttrMap m;
    m.reserve(static_cast<size_t>(n.attribute_size()));
    for (const auto& a : n.attribute()) {
        m.insert_or_assign(a.name(), ParseAttr(a));
    }
    return m;
}

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
            }) | std::ranges::to<std::vector<ir::ValueId>>(),

            ParseAttrs(onnx_node)
        });
    }

    return graph;
}

} // namespace tensor_compiler::frontend
