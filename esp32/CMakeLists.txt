# The following five lines of boilerplate have to be in your project's
# CMakeLists in this exact order for cmake to work correctly
cmake_minimum_required(VERSION 3.16)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(edgeml)

esptool_py_flash_to_partition(
    flash
    model_a
    "${CMAKE_CURRENT_SOURCE_DIR}/../model/lenet_quantized.tflite"
)
