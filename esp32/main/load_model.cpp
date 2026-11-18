#include "load_model.h"

#include <nvs.h>

const char* modelA_PartitionName = "model_a";
const char* modelB_PartitionName = "model_b";

const esp_partition_type_t ModelPartitionType = static_cast<esp_partition_type_t>(0x40);
const esp_partition_subtype_t ModelPartitionSubtype = ESP_PARTITION_SUBTYPE_ANY;


static const char* GetPartitionName(uint8_t modelIdx)
{
  const char* partitionName = modelA_PartitionName;
  if (modelIdx != 0)
    partitionName = modelB_PartitionName;

  return partitionName;
}

namespace Ryk
{
  ModelStorage::ModelStorage()
  {
  }

  ModelStorage::~ModelStorage()
  {
    if (hMem != 0)
      esp_partition_munmap(hMem);
  }

  int ModelStorage::GetCurrentModelIdx(uint8_t* currentModelIdx) const
  {
    *currentModelIdx = 0;
    nvs_handle_t hNvs = 0;
    esp_err_t err = nvs_open("model", NVS_READONLY, &hNvs);
    if (err == ESP_OK)
      err = nvs_get_u8(hNvs, "current", currentModelIdx);

    nvs_close(hNvs);

    if (err == ESP_ERR_NVS_NOT_FOUND)
      err = ESP_OK;

    return err;
  }

  esp_err_t ModelStorage::GetModelData(const uint8_t** modelData) const
  {
    esp_err_t err = ESP_OK;
    uint8_t currentModelIdx = 0;
    err = GetCurrentModelIdx(&currentModelIdx);
    if (err != ESP_OK)
      return err;

    const char* partitionName = GetPartitionName(currentModelIdx);
    const esp_partition_t* partition = esp_partition_find_first(ModelPartitionType, ModelPartitionSubtype, partitionName);
    if (partition == nullptr)
      return ESP_ERR_NOT_FOUND;

    const void* mem = nullptr;
    err = esp_partition_mmap(partition, 0, partition->size, ESP_PARTITION_MMAP_DATA, &mem, &hMem);
    if (err != ESP_OK)
      return err;

    *modelData = reinterpret_cast<const uint8_t*>(mem);

    return ESP_OK;
  }

  esp_err_t ModelStorage::UpdateModel(const uint8_t* modelData, size_t pos, size_t dataSize)
  {
    esp_err_t err = ESP_OK;
    uint8_t currentModelIdx = 0;
    err = GetCurrentModelIdx(&currentModelIdx);
    if (err != ESP_OK)
      return err;

    const uint8_t updatePartitionIdx = currentModelIdx == 0 ? 1 : 0;
    const char* partitionName = GetPartitionName(updatePartitionIdx);
    const esp_partition_t* partition = esp_partition_find_first(ModelPartitionType, ModelPartitionSubtype, partitionName);
    if (partition == nullptr)
      return ESP_ERR_NOT_FOUND;

    if (dataSize > partition->size)
      return ESP_ERR_INVALID_SIZE;

    err = esp_partition_erase_range(partition, 0, partition->size);
    if (err != ESP_OK)
      return err;

    err = esp_partition_write(partition, pos, modelData, dataSize);
    if (err != ESP_OK)
      return err;

    return ESP_OK;
  }

}
