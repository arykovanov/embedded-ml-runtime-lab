#include "ryk_wifi.h"

#include <esp_check.h>
#include <esp_err.h>
#include <esp_wifi.h>

#include <wifi_provisioning/scheme_ble.h>
#include <wifi_provisioning/manager.h>

static const char* TAG = "Wifi";

static void OnEvent(void* arg, esp_event_base_t base, int32_t id, void* data)
{
  esp_err_t err = ESP_OK;
  ESP_LOGI(TAG, "Event: %s, id: %" PRId32, base, id);
  if (base == IP_EVENT and id == IP_EVENT_STA_GOT_IP)
  {
    const auto* event = static_cast<const ip_event_got_ip_t*>(data);
    ESP_LOGI(TAG, "IP Address: " IPSTR, IP2STR(&event->ip_info.ip));
  }
  if (base == WIFI_EVENT && id == WIFI_EVENT_STA_START)
  {
    err = esp_wifi_connect();
    ESP_RETURN_VOID_ON_ERROR(err, TAG, "%i - Failed to connect to WiFi", err);
  }
}

esp_err_t Ryk::ConnectToWifi()
{
  esp_err_t err = ESP_OK;

  ESP_LOGI(TAG, "Initializing NetIf");
  err = esp_netif_init();
  ESP_RETURN_ON_ERROR(err, TAG, "%i - Failed to init netif", err);

  ESP_LOGI(TAG, "Creating event loop");
  err = esp_event_loop_create_default();
  ESP_RETURN_ON_ERROR(err, TAG, "%i - Failed to create event loop", err);

  err = esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, OnEvent, nullptr);
  ESP_RETURN_ON_ERROR(err, TAG, "%i - Failed to Register event WIFI_EVENT+ESP_EVENT_ANY_ID", err);
  err = esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP, OnEvent, nullptr);
  ESP_RETURN_ON_ERROR(err, TAG, "%i - Failed to Register event IP_EVENT+IP_EVENT_STA_GOT_IP", err);
  err = esp_event_handler_register(WIFI_PROV_EVENT, ESP_EVENT_ANY_ID, OnEvent, nullptr);
  ESP_RETURN_ON_ERROR(err, TAG, "%i - Failed to Register event WIFI_PROV_EVENT, ESP_EVENT_ANY_ID", err);

  ESP_LOGI(TAG, "Creating WiFi Station");
  esp_netif_t* netif = esp_netif_create_default_wifi_sta();

  ESP_LOGI(TAG, "Initializing Wifi");
  wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
  err = esp_wifi_init(&cfg);
  ESP_RETURN_ON_ERROR(err, TAG, "%i - Failed to init wifi", err);

  ESP_LOGI(TAG, "Check if WiFi provisioned");
  bool isProvisioned = false;
  err = wifi_prov_mgr_is_provisioned(&isProvisioned);
  ESP_RETURN_ON_ERROR(err, TAG, "%i - Failed to detect state of Wifi prosionioning", err);

  if (!isProvisioned)
  {
    ESP_LOGI(TAG, "WiFi not provisioned");
    wifi_prov_mgr_config_t config = {
        .scheme = wifi_prov_scheme_ble,
        .scheme_event_handler = WIFI_PROV_SCHEME_BLE_EVENT_HANDLER_FREE_BTDM
    };

    err = wifi_prov_mgr_init(config);
    ESP_RETURN_ON_ERROR(err, TAG, "%i Failed to init wifi provisioning manager", err);

    const char* serviceName = "ARM_PREP";
    const char* serviceKey = "qwerty";
    wifi_prov_security_t security = WIFI_PROV_SECURITY_1;
    const char* pop = "abcd1234";

    err = wifi_prov_mgr_start_provisioning(security, pop, serviceName, serviceKey);
    ESP_RETURN_ON_ERROR(err, TAG, "%i - Failed to start WiFi provisioning", err);

    ESP_LOGI(TAG, "Waiting for WiFi provisioning");
    wifi_prov_mgr_wait();

    ESP_LOGI(TAG, "WiFi provisioning finished");
    wifi_prov_mgr_deinit();
  }
  else
  {
    ESP_LOGI(TAG, "WiFi already provisioned");
    err = esp_wifi_set_mode(WIFI_MODE_STA);
    ESP_RETURN_ON_ERROR(err, TAG, "Failed to set STA mode");
    err = esp_wifi_start();
    ESP_RETURN_ON_ERROR(err, TAG, "Failed to start WiFi");
  }

  esp_netif_ip_info_t info;
  err = esp_netif_get_ip_info(netif, &info);
  if (err == ESP_OK)
    ESP_LOGI(TAG, "WiFi IP Address: " IPSTR, IP2STR(&info.ip));
  else
    ESP_LOGE(TAG, "%i - Failed to get IP Address.", err);

  return ESP_OK;
}
