#include "relu.h"

#include <gtest/gtest.h>
#include <filesystem>

extern int ArgC;
extern char** ArgV;


TEST(Operator, RykRelu)
{
  const std::string TestModelPath = std::filesystem::path(ArgV[0]).parent_path().append("../model/relu_test.onnx");

  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ReluTest");
  Ryk::OpDomain domain;
  Ort::SessionOptions options;
  options.Add(domain);

  std::vector<float> inputData(28*28, 10);
  inputData[0] = -10;
  std::vector<int64_t> inputShape = {1,28,28};
  Ort::AllocatorWithDefaultOptions allocator;
  Ort::Value inputTensor = Ort::Value::CreateTensor<float>(allocator, inputShape.data(), inputShape.size());
  std::copy(inputData.begin(), inputData.end(), inputTensor.GetTensorMutableData<float>());

  constexpr std::array<const char*, 1> sessInputNames{"images"};
  constexpr std::array<const char*, 1> sessOutputNames{"logits"};

  Ort::Session session(env, TestModelPath.c_str(), options);
  std::vector<Ort::Value> outputTensor = session.Run(
    Ort::RunOptions{nullptr},
    sessInputNames.data(), &inputTensor, sessInputNames.size(),
    sessOutputNames.data(), sessOutputNames.size()
  );


  ASSERT_EQ(1, outputTensor.size());
  Ort::TensorTypeAndShapeInfo typeInfo = outputTensor[0].GetTensorTypeAndShapeInfo();
  ASSERT_EQ(typeInfo.GetShape(), inputShape);

  std::vector<float> expectedOutputData = inputData;
  expectedOutputData[0] = 0;

  ASSERT_EQ(typeInfo.GetElementCount(), expectedOutputData.size());
  const float* outputTensorData = outputTensor[0].GetTensorData<float>();
  for (size_t i = 0; i < typeInfo.GetElementCount(); ++i)
  {
    ASSERT_EQ(expectedOutputData[i], outputTensorData[i]);
  }
}
