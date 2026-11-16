#pragma once

#include <vector>
#include <stdint.h>

namespace Ryk
{

  struct SavedTensor
  {
    std::vector<int64_t> Shape;
    std::vector<float> Data;
  };

  SavedTensor LoadSampleBatch();
  SavedTensor LoadSampleBatch(const uint8_t* dataStart, const uint8_t* dataEnd);
}