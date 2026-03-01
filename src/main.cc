#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "onnx_importer.hpp"
#include "onnx_loader.hpp"
#include "parse_cli.hpp"
#include "graph_dump.hpp"

namespace tc = tensor_compiler;

int main(int argc, const char** argv) {
    try {
        auto [program, cfg] = tc::cli::ParseCli(argc, argv);
        if (program.mode == tc::cli::CliMode::kExit) {
            std::cout << program.exit_action.exit_text;
            return program.exit_action.exit_code;
        }

        onnx::ModelProto model = tc::frontend::LoadOnnxModel(cfg.value().onnx_filename);
        tc::ir::Graph graph = tc::frontend::ImportOnnx(model.graph());

        std::ofstream os{cfg.value().dump_filename};
        os << tc::viz::GraphDump(graph);

        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}
