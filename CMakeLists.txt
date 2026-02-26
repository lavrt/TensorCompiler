cmake_minimum_required(VERSION 3.20)
project(TensorCompiler LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(protobuf CONFIG REQUIRED)
find_package(ONNX CONFIG REQUIRED)
find_package(Boost CONFIG REQUIRED)

add_executable(run
    src/main.cc
    src/frontend/onnx_loader.cc
    src/frontend/onnx_importer.cc
    src/cli/parse_cli.cc
    src/viz/graph_dump.cc
)

target_include_directories(run
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include/cli
        ${CMAKE_SOURCE_DIR}/include/frontend
        ${CMAKE_SOURCE_DIR}/include/ir
        ${CMAKE_SOURCE_DIR}/include/viz
)

target_link_libraries(run
    PRIVATE
        protobuf::libprotobuf
        onnx::onnx
        Boost::program_options
)
