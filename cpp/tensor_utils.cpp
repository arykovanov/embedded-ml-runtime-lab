#include "tensor_utils.h"

#include <fstream>
#include <print>

Ryk::SavedTensor Ryk::LoadSampleBatch()
{
  std::ifstream file("model/sample_images.bin", std::ios::binary);
  std::uint32_t rank = 0;
  file.read(reinterpret_cast<char*>(&rank), sizeof(rank));
  SavedTensor result;
  result.Shape.reserve(rank);
  result.Shape.resize(rank);
  file.read(reinterpret_cast<char*>(result.Shape.data()), result.Shape.size() * sizeof(std::int64_t));

  std::uint32_t dataSize = 0;
  file.read(reinterpret_cast<char*>(&dataSize), sizeof(dataSize));
  dataSize /= sizeof(float);
  result.Data.reserve(dataSize);
  result.Data.resize(dataSize);
  file.read(reinterpret_cast<char*>(result.Data.data()), result.Data.size() * sizeof(float));

  return result;
}

Ryk::SavedTensor Ryk::LoadSampleBatch(const uint8_t* dataStart, const uint8_t* dataEnd)
{
  std::uint32_t rank = *reinterpret_cast<const std::uint32_t*>(dataStart);
  dataStart += sizeof(std::uint32_t);
  if (dataStart >= dataEnd)
  {
    std::println("Invalid sample batch format");
    return {};
  }

  SavedTensor result;
  result.Shape.reserve(rank);
  result.Shape.resize(rank);
  const int64_t* shapeData = reinterpret_cast<const int64_t*>(dataStart);
  std::copy(shapeData, shapeData + rank, result.Shape.data());
  const size_t shapeDataSize = result.Shape.size() * sizeof(std::int64_t);
  dataStart += shapeDataSize;
  if (dataStart >= dataEnd)
  {
    std::println("Invalid sample batch format");
    return {};
  }

  const std::uint32_t dataSize = *reinterpret_cast<const std::uint32_t*>(dataStart);
  dataStart += sizeof(dataSize);
  if (dataStart >= dataEnd)
  {
    std::println("Invalid sample batch format");
    return {};
  }

  const std::uint32_t dataValuesNumber = dataSize / sizeof(float);
  result.Data.reserve(dataValuesNumber);
  result.Data.resize(dataValuesNumber);
  const float* tensorValues = reinterpret_cast<const float*>(dataStart);
  std::copy(tensorValues, tensorValues + dataValuesNumber, result.Data.data());

  return result;
}
