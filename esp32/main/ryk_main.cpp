#include "ryk_model.h"
#include "ryk_http_server.h"
#include "ryk_network.h"

#include <tensor_utils.h>

#include <esp_check.h>
#include <esp_heap_caps.h>
#include <esp_timer.h>
#include <nvs_flash.h>

static const char* TAG = "Ryk";

class NVSHolder
{
public:
  esp_err_t Init()
  {
    return nvs_flash_init();
  }

  ~NVSHolder()
  {
    nvs_flash_deinit();
  }
};

static Ryk::Model model;
static NVSHolder nvs;

extern "C"
void app_main(void)
{
  esp_err_t err = ESP_OK;

  err = nvs.Init();
  ESP_RETURN_VOID_ON_ERROR(err, TAG, "%i - Failed to init NVS", err);

  err = Ryk::StartNetwork();
  ESP_RETURN_VOID_ON_ERROR(err, TAG, "Failed to start network");

  err = model.Init();
  ESP_RETURN_VOID_ON_ERROR(err, TAG, "Failed to init model");

  err = Ryk::StartHttpServer(model);
  ESP_RETURN_VOID_ON_ERROR(err, TAG, "Failed to start HTTP server");
}
