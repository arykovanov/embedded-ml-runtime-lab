cmake_minimum_required(VERSION 3.28.3)

project(onnx_exporter)

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
)

target_link_libraries(onnx_exporter PRIVATE onnxruntime)
