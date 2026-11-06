#include <iostream>
#include <onnxruntime_cxx_api.h>

int main(int argc, char** argv) {
    std::cout << "Hello, ONNX Runtime v" << Ort::GetVersionString() << std::endl;
    return 0;
}
