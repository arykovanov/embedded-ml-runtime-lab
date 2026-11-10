cmake_minimum_required(VERSION 3.28.3)
set(CMAKE_CXX_COMPILER g++-14)

project(onnx_exporter)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES 
    IMPORTED_LOCATION
        "${CMAKE_CURRENT_LIST_DIR}/onnxruntime/lib/libonnxruntime.so"
    INTERFACE_INCLUDE_DIRECTORIES
        "${CMAKE_CURRENT_LIST_DIR}/onnxruntime/include"
)

add_executable(onnx_exporter)

target_sources(onnx_exporter
    PRIVATE
        main.cpp
        cpp/relu.cpp
)

target_include_directories(onnx_exporter
    PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/cpp"
)

target_link_libraries(onnx_exporter PRIVATE onnxruntime)
target_compile_features(onnx_exporter PRIVATE cxx_std_23)

target_compile_options(onnx_exporter PRIVATE
    $<$<CONFIG:Debug>:-O0>
    $<$<CONFIG:Debug>:-g>
)

add_executable(onnx_exporter_test)
target_link_libraries(onnx_exporter_test PRIVATE
    onnxruntime
    gtest
)
target_compile_features(onnx_exporter_test PRIVATE cxx_std_23)
target_compile_options(onnx_exporter_test PRIVATE
    $<$<CONFIG:Debug>:-O0>
    $<$<CONFIG:Debug>:-g>
)
target_sources(onnx_exporter_test
    PRIVATE
        cpp/relu.cpp
        cpp/relu_test.cpp
        cpp/relu_test_main.cpp
)
target_include_directories(onnx_exporter_test
    PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/cpp"
)

# add_custom_command(check COMMAND
#     $<TARGET_FILE:onnx_exporter_test>
# )
# add_custom_target(check ALL COMMAND
#     $<TARGET_FILE:onnx_exporter_test>
#     "${CMAKE_CURRENT_SOURCE_DIR}/onnx_exporter_test"
#     WORKING_DIRECTORY
#     "${CMAKE_CURRENT_SOURCE_DIR}"
#     USES_TERMINAL
#     VERBATIM
# )
# add_dependencies(check onnx_exporter_test)
