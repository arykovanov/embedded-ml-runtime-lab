#include "ryk_model_data.h"

#include <esp_check.h>
#include <nvs.h>

static const char* modelA_PartitionName = "model_a";
static const char* modelB_PartitionName = "model_b";
static const char* TAG = "Model Storage";

static const esp_partition_type_t ModelPartitionType = static_cast<esp_partition_type_t>(0x40);
static const esp_partition_subtype_t ModelPartitionSubtype = ESP_PARTITION_SUBTYPE_ANY;

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

  esp_err_t ModelStorage::GetCurrentModelIdx(uint8_t* currentModelIdx) const
  {
    *currentModelIdx = 0;
    nvs_handle_t hNvs = 0;
    esp_err_t err = nvs_open("model", NVS_READONLY, &hNvs);
    if (err == ESP_OK)
    {
      err = nvs_get_u8(hNvs, "current", currentModelIdx);
      nvs_close(hNvs);
    }

    if (err == ESP_ERR_NVS_NOT_FOUND)
      err = ESP_OK;

    ESP_RETURN_ON_ERROR(err, TAG, "Cannot read current model index from NVS");

    return err;
  }

  esp_err_t ModelStorage::SetCurrentModelIdx(uint8_t currentModelIdx)
  {
    nvs_handle_t hNvs = 0;
    esp_err_t err = nvs_open("model", NVS_READWRITE, &hNvs);
    ESP_RETURN_ON_ERROR(err, TAG, "Cannot open NVS");

    err = nvs_set_u8(hNvs, "current", currentModelIdx);

    nvs_close(hNvs);

    ESP_RETURN_ON_ERROR(err, TAG, "Failed to store current model index in NVS");

    return err;
  }

  esp_err_t ModelStorage::GetModelData(const uint8_t** modelData) const
  {
    ESP_LOGI(TAG, "Loading model data from partition");

    esp_err_t err = ESP_OK;
    uint8_t currentModelIdx = 0;
    err = GetCurrentModelIdx(&currentModelIdx);
    ESP_RETURN_ON_ERROR(err, TAG, "Failed to get current model index");

    const char* partitionName = GetPartitionName(currentModelIdx);
    ESP_LOGI(TAG, "Look for partition %s", partitionName);
    const esp_partition_t* partition = esp_partition_find_first(ModelPartitionType, ModelPartitionSubtype, partitionName);
    if (partition == nullptr)
      ESP_RETURN_ON_ERROR(ESP_ERR_NOT_FOUND, TAG, "Failed to Find model's partition %s", partitionName);

    ESP_LOGI(TAG, "Map parition '%s' into memory", partitionName);
    const void* mem = nullptr;
    err = esp_partition_mmap(partition, 0, partition->size, ESP_PARTITION_MMAP_DATA, &mem, &hMem);
    ESP_RETURN_ON_ERROR(err, TAG, "Failed to mmap partition %s", partitionName);

    ESP_LOGI(TAG, "Model data mapped into memory");

    *modelData = reinterpret_cast<const uint8_t*>(mem);

    return ESP_OK;
  }

  esp_err_t ModelStorage::UpdateModel(const uint8_t* modelData, size_t pos, size_t dataSize)
  {
    ESP_LOGI(TAG, "Writing a new model to flash");
    esp_err_t err = ESP_OK;
    uint8_t currentModelIdx = 0;
    err = GetCurrentModelIdx(&currentModelIdx);
    ESP_RETURN_ON_ERROR(err, TAG, "Failed to get current model index");
    ESP_LOGI(TAG, "Current model index - %i", (int)currentModelIdx);

    const uint8_t updatePartitionIdx = currentModelIdx == 0 ? 1 : 0;
    ESP_LOGI(TAG, "Model index to update - %i", (int)updatePartitionIdx);

    const char* partitionName = GetPartitionName(updatePartitionIdx);
    ESP_LOGI(TAG, "Model will be written to partition '%s'", partitionName);

    const esp_partition_t* partition = esp_partition_find_first(ModelPartitionType, ModelPartitionSubtype, partitionName);

    if (partition == nullptr)
      ESP_RETURN_ON_ERROR(ESP_ERR_NOT_FOUND, TAG, "Failed to Find model's partition %s", partitionName);

    if (dataSize > partition->size)
      ESP_RETURN_ON_ERROR(ESP_ERR_INVALID_SIZE, TAG, "Model's partition %s too small", partitionName);

    // err = esp_partition_erase_range(partition, pos, dataSize);
    // ESP_RETURN_ON_ERROR();

    ESP_LOGI(TAG, "Writing model data to flash...");

    err = esp_partition_write(partition, pos, modelData, dataSize);
    ESP_RETURN_ON_ERROR(err, TAG, "Failed to write model data at position %zu", pos);

    err = SetCurrentModelIdx(updatePartitionIdx);
    ESP_RETURN_ON_ERROR(err, TAG, "Failed to store current model index into NVS");

    ESP_LOGI(TAG, "Model data sucessfully written flash.");

    return ESP_OK;
  }
}
