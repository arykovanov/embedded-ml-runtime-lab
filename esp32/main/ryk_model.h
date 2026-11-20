#pragma once

#include "ryk_model_data.h"

#include <esp_err.h>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

#include <stdint.h>
#include <memory>
#include <span>
#include <vector>


namespace Ryk
{
  class Model
  {
  public:
    Model();
    ~Model();

    esp_err_t Init();

    esp_err_t Execute(std::span<const float> image, std::vector<float>& logits);
    esp_err_t UpdateModel(const char* modelData, std::size_t dataSize);

  private:
    esp_err_t Deinit();

  private:
    Ryk::ModelStorage modelLoader;
    const std::size_t TensorArenaSize = 19 * 1024;
    uint8_t TensorArena[19 * 1024] = {};
    std::unique_ptr<tflite::MicroMutableOpResolver<10>> resolver;
    std::unique_ptr<tflite::MicroInterpreter> interpreter;
  };
}
