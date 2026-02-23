#include "parse_cli.hpp"

#include <sstream>
#include <string>

#include <boost/program_options.hpp>

namespace tensor_compiler {

std::pair<CliResult, std::optional<ProgramConfig>> ParseCli(int argc, const char** argv) {
    namespace po = boost::program_options;
    
    po::options_description desc("Options");
    desc.add_options()
        (
            "help,h",
            "show this help and exit"
        )
        (
            "onnx,o",
            po::value<std::string>()->required(),
            "path to onnx file"
        );
    
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    std::ostringstream help_text;
    help_text << desc;

    if (vm.count("help")) {
        return {
            CliResult {
                .mode = CliMode::kExit,
                .exit_action {
                    .exit_code = 0,
                    .exit_text = help_text.str()
                }
            },
            std::nullopt
        };
    }

    return {
        CliResult{},
        ProgramConfig{
            .onnx_filename = vm["onnx"].as<std::string>()
        }
    };
}

} // namespace tensor_compiler
