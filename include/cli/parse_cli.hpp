#pragma once

#include <optional>
#include <string>

namespace tensor_compiler::cli {

enum class CliMode {
    kRun = 0,
    kExit = 1,
};

struct ExitAction {
    int exit_code;
    std::string exit_text;
};

struct CliResult {
    CliMode mode = CliMode::kRun;
    ExitAction exit_action;
};

struct ProgramConfig {
    std::string onnx_filename;
};

std::pair<CliResult, std::optional<ProgramConfig>> ParseCli(int argc, const char** argv);

} // namespace tensor_compiler::cli
