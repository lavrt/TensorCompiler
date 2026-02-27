#include "graph_dump.hpp"

#include <format>
#include <iomanip>
#include <sstream>
#include <string>

namespace tensor_compiler::viz {

std::string GraphDump(const ir::Graph& g) {
    std::stringstream ss;

    ss << "digraph G {\n"
       << "  rankdir=LR;\n"
       << "  node [shape=box, fontname=\"Helvetica\"];\n"
       << "  edge [fontname=\"Helvetica\"];\n\n";
    
    for (size_t i = 0; i != g.nodes.size(); ++i) {
        const auto& n = g.nodes[i];

        std::string label = n.op_name;
        if (!n.name.empty()) {
            label += (" \\n " + n.name);
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
