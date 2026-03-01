cmake_minimum_required(VERSION 3.11)
project(TensorCompiler LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(protobuf CONFIG REQUIRED)
find_package(ONNX CONFIG REQUIRED)
find_package(Boost CONFIG REQUIRED)

# tc_core -----------------------------------

add_library(tc_core
    src/frontend/onnx_loader.cc
    src/frontend/onnx_importer.cc
    src/viz/graph_dump.cc
)

target_include_directories(tc_core
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include/frontend
        ${CMAKE_SOURCE_DIR}/include/ir
        ${CMAKE_SOURCE_DIR}/include/viz
)

target_link_libraries(tc_core
    PUBLIC
        protobuf::libprotobuf
        onnx::onnx
)

# program -----------------------------------

add_executable(run
    src/main.cc
    src/cli/parse_cli.cc
)

target_include_directories(run
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include/cli
)

target_link_libraries(run
    PRIVATE
        tc_core
        Boost::program_options
)

# tests -------------------------------------

include(CTest)
if (BUILD_TESTING)
    add_subdirectory(tests)
endif()
