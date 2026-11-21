#include "ryk_http_server.h"
#include "ryk_model.h"

#include <esp_check.h>
#include <esp_http_server.h>

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>
#include <sstream>


static const char* TAG = "HTTP";
static const size_t MAX_BODY_SIZE = 16384;


static esp_err_t GetRootFunc(httpd_req_t* req)
{
  httpd_resp_send(req, "Hello", HTTPD_RESP_USE_STRLEN);
  return ESP_OK;
}

static esp_err_t GetContentLength(httpd_req_t* req, size_t* contentLength)
{
  if (req == nullptr || contentLength == nullptr)
    return ESP_ERR_INVALID_ARG;

  const size_t headerLen = httpd_req_get_hdr_value_len(req, "Content-Length");
  if (headerLen == 0)
    return ESP_ERR_INVALID_SIZE;

  std::string headerValue(headerLen + 1, '\0');
  esp_err_t err = httpd_req_get_hdr_value_str(req, "Content-Length", &headerValue[0], headerValue.size());
  ESP_RETURN_ON_ERROR(err, TAG, "%i - Failed to get 'Content-Length'", err);

  size_t bodySize = 0;
  const char* end = headerValue.data() + headerLen;
  const auto result = std::from_chars(headerValue.data(), end, bodySize);
  if (result.ec != std::errc{} || result.ptr != end)
    return ESP_ERR_INVALID_SIZE;

  *contentLength = bodySize;
  return ESP_OK;
}

static esp_err_t ReceiveBody(httpd_req_t* req, std::string* body, size_t maxSize)
{
  if (req == nullptr || body == nullptr)
    return ESP_ERR_INVALID_ARG;

  body->clear();
  size_t contentLength = 0;
  esp_err_t err = GetContentLength(req, &contentLength);
  ESP_RETURN_ON_ERROR(err, TAG, "%i - Failed to get content length", err);

  if (contentLength > maxSize)
    return ESP_ERR_INVALID_SIZE;

  body->resize(contentLength);
  size_t receivedSize = 0;
  while (receivedSize < contentLength)
  {
    const size_t readSize = std::min(contentLength - receivedSize,
                                   static_cast<size_t>(std::numeric_limits<int>::max()));
    const int received = httpd_req_recv(req, reinterpret_cast<char*>(body->data() + receivedSize), readSize);
    if (received <= 0)
    {
      body->clear();
      return received == HTTPD_SOCK_ERR_TIMEOUT ? ESP_ERR_TIMEOUT : ESP_FAIL;
    }

    receivedSize += static_cast<size_t>(received);
  }

  return ESP_OK;
}

static esp_err_t SendBody(httpd_req_t* req, const uint8_t* data, size_t dataSize)
{
  if (req == nullptr || (data == nullptr && dataSize != 0))
    return ESP_ERR_INVALID_ARG;

  if (dataSize > static_cast<size_t>(std::numeric_limits<ssize_t>::max()))
    return ESP_ERR_INVALID_SIZE;

  return httpd_resp_send(req, reinterpret_cast<const char*>(data), static_cast<ssize_t>(dataSize));
}

static esp_err_t PostRootFunc(httpd_req_t* req)
{
  ESP_RETURN_ON_ERROR(httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*"), TAG, "Failed to set CORS header");

  Ryk::Model* model = static_cast<Ryk::Model*>(req->user_ctx);
  std::string body;
  esp_err_t err = ReceiveBody(req, &body, MAX_BODY_SIZE);
  if (err == ESP_ERR_INVALID_SIZE)
  {
    httpd_resp_set_status(req, HTTPD_400);
    httpd_resp_send(req, "Invalid 'Content-Length'", HTTPD_RESP_USE_STRLEN);
    return ESP_OK;
  }

  ESP_RETURN_ON_ERROR(err, TAG, "%i - Failed to receive body", err);

  std::istringstream in(body);
  std::size_t elementsNumber = 0;
  in >> elementsNumber;
  std::vector<float> image;
  for (size_t i = 0; i < elementsNumber; ++i)
  {
    float pixel = 0;
    if(!(in >> pixel))
    {
      httpd_resp_set_status(req, HTTPD_400);
      httpd_resp_send(req, "Invalid Pixel", HTTPD_RESP_USE_STRLEN);
      return ESP_OK;
    }

    image.push_back(pixel);
  }

  std::vector<float> logits;
  err = model->Execute(image, logits);
  if (err == ESP_ERR_INVALID_ARG)
  {
    httpd_resp_set_status(req, HTTPD_400);
    httpd_resp_send(req, "Invalid input image", HTTPD_RESP_USE_STRLEN);
    return ESP_OK;
  }
  else if (err != ESP_OK)
  {
    httpd_resp_set_status(req, HTTPD_500);
    httpd_resp_send(req, "Internal Error", HTTPD_RESP_USE_STRLEN);
    return ESP_OK;
  }

  std::stringstream out;
  out << logits.size();
  for (const auto logit : logits)
    out << " " << logit;

  const auto& str = out.view();
  return SendBody(req, reinterpret_cast<const uint8_t*>(str.data()), str.size());
}

static esp_err_t PostModelFunc(httpd_req_t* req)
{
  Ryk::Model* model = static_cast<Ryk::Model*>(req->user_ctx);
  std::string body;
  esp_err_t err = ReceiveBody(req, &body, 1024*1024);
  if (err == ESP_ERR_INVALID_SIZE)
  {
    httpd_resp_set_status(req, HTTPD_400);
    httpd_resp_send(req, "Invalid 'Content-Length'", HTTPD_RESP_USE_STRLEN);
    return ESP_OK;
  }

  ESP_RETURN_ON_ERROR(err, TAG, "%i - Failed to receive body", err);

  err = model->UpdateModel(body.data(), body.size());
  if (err == ESP_ERR_INVALID_ARG)
  {
    httpd_resp_set_status(req, HTTPD_400);
    httpd_resp_send(req, "Invalid input image", HTTPD_RESP_USE_STRLEN);
    return ESP_OK;
  }
  else if (err != ESP_OK)
  {
    httpd_resp_set_status(req, HTTPD_500);
    httpd_resp_send(req, "Internal Error", HTTPD_RESP_USE_STRLEN);
    return ESP_OK;
  }

  httpd_resp_send(req, "", 0);

  return ESP_OK;
}

esp_err_t Ryk::StartHttpServer(Model& model)
{
  httpd_handle_t hd = nullptr;
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  esp_err_t err = ESP_OK;

  err = httpd_start(&hd, &config);
  if (err != ESP_OK)
    ESP_LOGE(TAG, "Failed to start HTTP server.");


  if (err == ESP_OK)
  {
    static const httpd_uri_t GetRootHandler =
    {
      .uri = "/",
      .method = HTTP_GET,
      .handler = GetRootFunc,
      .user_ctx = nullptr
    };

    err = httpd_register_uri_handler(hd, &GetRootHandler);
    if (err != ESP_OK)
      ESP_LOGE(TAG, "Failed to register handler GET '%s'", GetRootHandler.uri);
  }


  if (err == ESP_OK)
  {
    static const httpd_uri_t PostRootHandler =
    {
      .uri = "/",
      .method = HTTP_POST,
      .handler = PostRootFunc,
      .user_ctx = &model
    };

    err = httpd_register_uri_handler(hd, &PostRootHandler);

    if (err != ESP_OK)
      ESP_LOGE(TAG, "Failed to register handler POST '%s'", PostRootHandler.uri);
  }


  if (err == ESP_OK)
  {
    static const httpd_uri_t PostModelHandler =
    {
      .uri = "/model",
      .method = HTTP_POST,
      .handler = PostModelFunc,
      .user_ctx = &model
    };

    err = httpd_register_uri_handler(hd, &PostModelHandler);

    if (err != ESP_OK)
      ESP_LOGE(TAG, "Failed to register handler POST '%s'", PostModelHandler.uri);
  }

  if (err != ESP_OK && hd != nullptr)
    httpd_stop(hd);

  return err;
}
