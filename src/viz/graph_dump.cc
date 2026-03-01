#include "graph_dump.hpp"

#include <format>
#include <iomanip>
#include <optional>
#include <sstream>
#include <string>

namespace {

std::string ToString(const tensor_compiler::ir::AttrValue& v) {
    using namespace tensor_compiler::ir;

    if (const auto* p = std::get_if<std::int64_t>(&v)) {
        return std::to_string(*p);
    }
    if (const auto* p = std::get_if<double>(&v)) {
        return std::format("{}", *p);
    }
    if (const auto* p = std::get_if<std::string>(&v)) {
        return *p;
    }
    if (const auto* p = std::get_if<std::vector<std::int64_t>>(&v)) {
        std::string s = "[";
        for (size_t i = 0; i < p->size(); ++i) {
            s += std::to_string((*p)[i]);
            if (i + 1 != p->size()) s += ", ";
        }
        s += "]";
        return s;
    }
    if (const auto* p = std::get_if<std::vector<double>>(&v)) {
        std::string s = "[";
        for (size_t i = 0; i < p->size(); ++i) {
            s += std::format("{}", (*p)[i]);
            if (i + 1 != p->size()) s += ", ";
        }
        s += "]";
        return s;
    }

    return "?";
}

std::string ToString(tensor_compiler::ir::DType dt) {
    namespace tc = tensor_compiler;
    switch (dt) {
        case tc::ir::DType::kF32: return "float32";
        case tc::ir::DType::kI64: return "int64";
        case tc::ir::DType::kI32: return "int32";
        case tc::ir::DType::kI8:  return "int8";
        case tc::ir::DType::kU8:  return "uint8";
        default: return "";
    }
}

} // namespace

namespace tensor_compiler::viz {

std::string GraphDump(const ir::Graph& g) {
    std::stringstream ss;

    ss << "digraph G {\n"
       << "  rankdir=TB;\n"
       << "  node [shape=box, fontname=\"Helvetica\"];\n"
       << "  edge [fontname=\"Helvetica\"];\n\n";
    
    for (size_t i = 0; i != g.nodes.size(); ++i) {
        const auto& n = g.nodes[i];

        std::string label = n.op_name;
        if (!n.name.empty()) {
            label += std::format(" ({})", n.name);
        }

        for (const auto& in : n.inputs) {
            const auto& v = g.values[in];
            if (v.is_initializer) {
                const auto& td = g.constants.at(in);
                label += "\\n" + v.name + " <";
                for (size_t i = 0, ie = td.shape.size(); i != ie; ++i) {
                    label += std::to_string(td.shape[i]);
                    if (i != ie - 1) {
                        label += ", ";
                    }
                }
                label += ">, " + ToString(td.dtype);
            }
        }

        if (!n.attrs.empty()) {
            for (const auto& [k, v] : n.attrs) {
                label += "\\n" + k + "=" + ToString(v);
            }
        }

        ss << std::format("  n{} ", i) << "[label=\"" << label << "\"];\n";
    }
    ss << "\n";

    for (size_t i = 0; i != g.values.size(); ++i) {
        const auto& v = g.values[i];

        if (!v.is_graph_input) {
            continue;
        }

        ss << std::format("  in{} ", i)
           << "[shape=oval, label=" << std::quoted(v.name) << "];\n";
    }
    ss << "\n";

    for (size_t i = 0; i != g.values.size(); ++i) {
        const auto& v = g.values[i];

        if (!v.is_graph_output) {
            continue;
        }

        ss << std::format("  out{} ", i)
           << "[shape=oval, label=" << std::quoted(v.name) << "];\n";
    }
    ss << "\n";

    for (size_t i = 0; i != g.values.size(); ++i) {
        const auto& v = g.values[i];

        if (v.is_graph_input) {
            for (auto&& c : v.consumers) {
                ss << std::format("  in{} -> n{} ", i, c)
                   << "[label=" << std::quoted(v.name) << "];\n";
            }
        }

        if (v.producer.has_value()) {
            for (auto&& c : v.consumers) {
                ss << std::format("  n{} -> n{} ", *v.producer, c)
                   << "[label=" << std::quoted(v.name) << "];\n";
            }
        }

        if (v.is_graph_output) {
            if (v.producer.has_value()) {
                ss << std::format("  n{} -> out{} ", *v.producer, i)
                   << "[label=" << std::quoted(v.name) << "];\n";
            } else if (v.is_graph_input) {
                ss << std::format("  in{} -> out{} ", i, i)
                   << "[label=" << std::quoted(v.name) << "];\n";
            }
        }
    }
    ss << "}\n";

    return ss.str();
}

} // namespace tensor_compiler::viz
