#include "cpp/relu.h"
#include "cpp/tensor_utils.h"

#include <onnxruntime_cxx_api.h>
#include <onnxruntime_lite_custom_op.h>

#include <algorithm>
#include <numeric>
#include <print>
#include <fstream>


void print_help()
{
  std::println("onnx --help");
  std::println("onnx <model_path>");
}

void saveLogits(const std::vector<int64_t>& shape, const std::vector<float>& logitsData)
{
  std::ofstream file("model/cpp_logits.bin", std::ios::binary);
  const int32_t rank = shape.size();
  file.write(reinterpret_cast<const char*>(&rank), sizeof(int32_t));
  file.write(reinterpret_cast<const char*>(shape.data()), shape.size() * sizeof(int64_t));

  const int32_t dataSize = logitsData.size() * sizeof(float);
  file.write(reinterpret_cast<const char*>(&dataSize), sizeof(int32_t));
  file.write(reinterpret_cast<const char*>(logitsData.data()), dataSize);
}

int main(int argc, char** argv)
{
  try
  {
    if (argc != 2)
    {
      std::println("Invalid parameters");
    }

    if (argv[1] == std::string_view("--help"))
    {
      print_help();
      return -1;
    }


    std::print("ONNX Runtime v{}\n", Ort::GetVersionString());
    std::print("Available providers:\n");
    for (const auto& provider : Ort::GetAvailableProviders())
    {
      std::print("  {}\n", provider);
    }

    Ryk::SavedTensor inputData = Ryk::LoadSampleBatch();
    std::vector<int64_t> imageShape = inputData.Shape;
    imageShape[0] = 1;
    const int64_t imageDataLen = std::accumulate(
      imageShape.begin(), imageShape.end(), 1,
      std::multiplies<int64_t>()
    );

    Ryk::OpDomain rykDomain;

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "Fashion MNIST");
    OrtApi api = Ort::GetApi();
    Ort::SessionOptions options;
    options.Add(rykDomain);

    Ort::Session session(env, argv[1], options);
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    const char* inputNames[] = {"images"};
    const char* outputNames[] = {"logits"};

    std::println("Running inference");


    std::vector<float> logits;
    std::vector<int64_t> logitsShape = {inputData.Shape[0],10};

    for (size_t pos = 0; pos < inputData.Data.size(); pos += imageDataLen)
    {
      Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo,
        inputData.Data.data() + pos,
        imageDataLen,
        imageShape.data(),
        imageShape.size()
      );

      std::vector<Ort::Value> outputTensors = session.Run(
        Ort::RunOptions{nullptr},
        inputNames, &inputTensor, 1, // number of inputs
        outputNames, 1 // number of outputs
      );

      for (const auto& value : outputTensors)
      {
        const Ort::TensorTypeAndShapeInfo tensorInfo = value.GetTensorTypeAndShapeInfo();
        const int64_t elementCount = tensorInfo.GetElementCount();
        const float* tensorValues = value.GetTensorData<float>();
        logits.insert(logits.end(), tensorValues, tensorValues + elementCount);
        std::print("[");
        for (int64_t i = 0; i < elementCount; ++i)
        {
          std::print("{}", tensorValues[i]);
          if ((i + 1)== elementCount)
          {
            std::print("{}", tensorValues[i]);
          }
          else
          {
            std::print("{},", tensorValues[i]);
          }
        }
        std::println("]");
      }

      saveLogits(logitsShape, logits);
    }
  }
  catch (const Ort::Exception& exc)
  {
    std::println("Exception: {}", exc.what());
  }

  return 0;
}
