#include "ryk_network.h"

#include <esp_check.h>
#include <esp_err.h>
#include <esp_wifi.h>
#include <esp_netif.h>
#include <esp_eth.h>
#include <soc/soc.h>
#include <soc/syscon_reg.h>

#include <wifi_provisioning/scheme_ble.h>
#include <wifi_provisioning/manager.h>

static const char* TAG = "Wifi";

static bool IsInQemu()
{
  unsigned firmware = REG_READ(SYSCON_DATE_REG - 4);
  ESP_LOGI(TAG, "QEMU signature: %u", firmware);
  return firmware == 0x51454D55u;
}

static void OnEvent(void* arg, esp_event_base_t base, int32_t id, void* data)
{
  esp_err_t err = ESP_OK;
  ESP_LOGI(TAG, "Event: %s, id: %" PRId32, base, id);
  if (base == IP_EVENT && (id == IP_EVENT_STA_GOT_IP || id == IP_EVENT_ETH_GOT_IP))
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

static esp_err_t ConnectToWifi()
{
  esp_err_t err = ESP_OK;
  ESP_LOGI(TAG, "Creating WiFi Station");
  esp_netif_t* netif = esp_netif_create_default_wifi_sta();

  ESP_LOGI(TAG, "Registering network events");
  err = esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, OnEvent, nullptr);
  ESP_RETURN_ON_ERROR(err, TAG, "%i - Failed to Register event WIFI_EVENT+ESP_EVENT_ANY_ID", err);
  err = esp_event_handler_register(WIFI_PROV_EVENT, ESP_EVENT_ANY_ID, OnEvent, nullptr);
  ESP_RETURN_ON_ERROR(err, TAG, "%i - Failed to Register event WIFI_PROV_EVENT, ESP_EVENT_ANY_ID", err);
  ESP_LOGI(TAG, "Registering Event handler for obtaining IP address");
  err = esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP, OnEvent, nullptr);
  ESP_RETURN_ON_ERROR(err, TAG, "%i - Failed to Register event IP_EVENT+IP_EVENT_STA_GOT_IP", err);

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
    wifi_prov_mgr_config_t config = {};
    config.scheme = wifi_prov_scheme_ble;
    config.scheme_event_handler = WIFI_PROV_SCHEME_BLE_EVENT_HANDLER_FREE_BTDM;

    ESP_LOGI(TAG, "Initializing WiFi provisioning manager");

    err = wifi_prov_mgr_init(config);
    ESP_RETURN_ON_ERROR(err, TAG, "%i Failed to init wifi provisioning manager", err);

    const char* serviceName = "ARM_PREP";
    const char* serviceKey = "qwerty";
    wifi_prov_security_t security = WIFI_PROV_SECURITY_1;
    const char* pop = "abcd1234";

    ESP_LOGI(TAG, "Starting WiFi provisioning");
    err = wifi_prov_mgr_start_provisioning(security, pop, serviceName, serviceKey);
    ESP_RETURN_ON_ERROR(err, TAG, "%i - Failed to start WiFi provisioning", err);

    ESP_LOGI(TAG, "Waiting for WiFi provisioning finish....");
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

esp_err_t StartEthernet()
{
  esp_err_t err = ESP_OK;
  esp_netif_config_t config = ESP_NETIF_DEFAULT_ETH();
  esp_netif_t* netif = esp_netif_new(&config);
  if (netif == nullptr)
  {
    ESP_LOGE(TAG, "Failed to create netif");
    return ESP_FAIL;
  }

  eth_mac_config_t macConfig = ETH_MAC_DEFAULT_CONFIG();
  esp_eth_mac_t* mac = esp_eth_mac_new_openeth(&macConfig);
  if (mac == nullptr)
  {
    ESP_LOGE(TAG, "Failed to create mac");
    return ESP_FAIL;
  }

  eth_phy_config_t phyConfig = ETH_PHY_DEFAULT_CONFIG();
  phyConfig.reset_gpio_num = -1;
  esp_eth_phy_t* phy = esp_eth_phy_new_dp83848(&phyConfig);
  if (phy == nullptr)
  {
    ESP_LOGE(TAG, "Failed to create phy");
    return ESP_FAIL;
  }

  esp_eth_config_t ethConfig = ETH_DEFAULT_CONFIG(mac, phy);
  esp_eth_handle_t hEth = nullptr;
  err = esp_eth_driver_install(&ethConfig, &hEth);
  ESP_RETURN_ON_ERROR(err, TAG, "Failed to create install eth driver");

  esp_eth_netif_glue_handle_t hNetifGlue = esp_eth_new_netif_glue(hEth);
  err = esp_netif_attach(netif, hNetifGlue);
  ESP_RETURN_ON_ERROR(err, TAG, "Failed to attach netif");

  ESP_LOGI(TAG, "Registering Event handler for obtaining IP address");
  err = esp_event_handler_register(IP_EVENT, IP_EVENT_ETH_GOT_IP, OnEvent, nullptr);
  ESP_RETURN_ON_ERROR(err, TAG, "%i - Failed to Register event IP_EVENT+IP_EVENT_STA_GOT_IP", err);

  err = esp_eth_start(hEth);
  ESP_RETURN_ON_ERROR(err, TAG, "%i - Failed to start ethernet", err);

  return err;
}

esp_err_t Ryk::StartNetwork()
{
  esp_err_t err = ESP_OK;

  ESP_LOGI(TAG, "Initializing NetIf");
  err = esp_netif_init();
  ESP_RETURN_ON_ERROR(err, TAG, "%i - Failed to init netif", err);

  ESP_LOGI(TAG, "Creating event loop");
  err = esp_event_loop_create_default();
  ESP_RETURN_ON_ERROR(err, TAG, "%i - Failed to create event loop", err);

  if (IsInQemu())
  {
    ESP_LOGI(TAG, "QEMU Environment. Starting Ethernet");
    err = StartEthernet();
    ESP_RETURN_ON_ERROR(err, TAG, "Failed to start Ethernet");
  }
  else
  {
    ESP_LOGI(TAG, "Running on real ESP32 board. Start WiFi");
    err = ConnectToWifi();
    ESP_RETURN_ON_ERROR(err, TAG, "Failed to connect to WiFi");
  }
  return ESP_OK;
}