cmake_minimum_required(VERSION 3.28.3)

project(onnx_exporter)

add_executable(onnx_exporter)

target_sources(onnx_exporter
    PRIVATE
        main.cpp
)