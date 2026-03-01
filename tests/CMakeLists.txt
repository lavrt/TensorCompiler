find_package(GTest CONFIG REQUIRED)

add_executable(tc_tests
    ${CMAKE_CURRENT_SOURCE_DIR}/test_graph.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/test_importer.cc
)

target_link_libraries(tc_tests
    PRIVATE
        tc_core
        GTest::gtest
        GTest::gtest_main
)

target_compile_definitions(tc_tests
    PRIVATE
        TC_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)

include(GoogleTest)
gtest_discover_tests(tc_tests
    DISCOVERY_TIMEOUT 30
)
