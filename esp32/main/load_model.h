#pragma once

#include <stdint.h>

#include <esp_err.h>
#include <esp_partition.h>

namespace Ryk
{

  class ModelStorage
  {
  public:
    ModelStorage();
    ~ModelStorage();

    esp_err_t GetModelData(const uint8_t** modelData) const;
    esp_err_t UpdateModel(const uint8_t* modelData);

  private:
    int GetCurrentModelIdx(uint8_t* currentModelIdx) const;

  private:
    mutable esp_partition_mmap_handle_t hMem = 0;
  };
}
