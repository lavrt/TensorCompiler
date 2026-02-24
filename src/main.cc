#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "onnx_loader.hpp"
#include "parse_cli.hpp"

static void PrintGraphSummary(const onnx::ModelProto& model) {
    const auto& g = model.graph();

    std::cout << "Graph name: " << g.name() << "\n";
    std::cout << "Inputs: " << g.input_size()
              << " Outputs: " << g.output_size()
              << " Initializers: " << g.initializer_size()
              << " Nodes: " << g.node_size() << "\n\n";

    for (int i = 0; i < g.node_size(); ++i) {
        const auto& n = g.node(i);

        std::cout << "Node #" << i << ": op_type=" << n.op_type();
        if (!n.name().empty()) std::cout << " name=" << n.name();
        std::cout << "\n  inputs: ";
        for (const auto& in : n.input()) std::cout << in << " ";
        std::cout << "\n  outputs: ";
        for (const auto& out : n.output()) std::cout << out << " ";
        std::cout << "\n";

        for (const auto& a : n.attribute()) {
            std::cout << "  attr: " << a.name() << " type=" << a.type() << "\n";
        }
        std::cout << "\n";
    }
}

namespace tc = tensor_compiler;

int main(int argc, const char** argv) {
    try {
        auto [program, cfg] = tc::cli::ParseCli(argc, argv);
        if (program.mode == tc::cli::CliMode::kExit) {
            std::cout << program.exit_action.exit_text;
            return program.exit_action.exit_code;
        }

        onnx::ModelProto model = tc::frontend::LoadOnnxModel(cfg.value().onnx_filename);

        PrintGraphSummary(model);

        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}
