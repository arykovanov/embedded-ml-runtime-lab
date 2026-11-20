#include "ryk_model.h"

#include <esp_check.h>

static const char* TAG = "MODEL";

static const tflite::Model* ParseTfLiteModel(const uint8_t* modelData)
{
  const tflite::Model* model = tflite::GetModel(modelData);
  if (model == nullptr)
  {
    ESP_LOGE(TAG, "Failed to parse model");
    return nullptr;
  }

  if (model->version() != TFLITE_SCHEMA_VERSION)
  {
    ESP_LOGE(TAG, "Unsupportd model version %i, required %i", (int)model->version(), (int)TFLITE_SCHEMA_VERSION);
    return nullptr;
  }

  return model;
}


using TfliteOpResolver = tflite::MicroMutableOpResolver<10>;

static std::unique_ptr<TfliteOpResolver> CreateOpResolver()
{
  auto resolver = std::make_unique<TfliteOpResolver>();
  resolver->AddConv2D();
  resolver->AddAveragePool2D();
  resolver->AddFullyConnected();
  resolver->AddReshape();
  resolver->AddPad();
  resolver->AddDepthwiseConv2D();
  resolver->AddTranspose();
  resolver->AddLogistic();

  return resolver;
}

static std::unique_ptr<tflite::MicroInterpreter> CreateInterpreter(
  const tflite::Model* model,
  TfliteOpResolver& resolver,
  uint8_t* arena,
  std::size_t arenaSize
)
{
  auto interpreter = std::make_unique<tflite::MicroInterpreter>(
    model,
    resolver,
    arena,
    arenaSize
  );

  TfLiteStatus status = interpreter->AllocateTensors();
  if (status != kTfLiteOk)
  {
    ESP_LOGE(TAG, "Failed to allocate tensors");
    return {};
  }

  return interpreter;
}

namespace Ryk
{
  Model::Model()
  {
  }

  esp_err_t Model::Init()
  {
    const uint8_t* modelData = nullptr;
    esp_err_t err = modelLoader.GetModelData(&modelData);
    if (err != ESP_OK)
      return err;

    const tflite::Model* model = ParseTfLiteModel(modelData);
    if (!model)
      return ESP_FAIL;

    resolver = CreateOpResolver();
    if (!resolver)
      return ESP_FAIL;

    interpreter = CreateInterpreter(model, *resolver, TensorArena, TensorArenaSize);
    if (!interpreter)
      return ESP_FAIL;

    return ESP_OK;
  }

  esp_err_t Model::Deinit()
  {
    resolver.reset();
    interpreter.reset();
    return ESP_OK;
  }

  Model::~Model()
  {
  }

  esp_err_t Model::Execute(std::span<const float> image, std::vector<float>& logits)
  {
    TfLiteTensor* input = interpreter->input(0);
    const TfLiteTensor* output = interpreter->output(0);
    const int32_t zeroPoint = input->params.zero_point;
    const float scale = input->params.scale;

    if (image.size() != 28*28)
      return ESP_ERR_INVALID_ARG;

    for (size_t qpos = 0; qpos < image.size(); ++qpos)
    {
      const int8_t q = static_cast<int8_t>(std::clamp(std::round(image[qpos] / scale + zeroPoint), -128.0f, 127.0f));
      input->data.int8[qpos] = q;
    }

    if (interpreter->Invoke() != kTfLiteOk)
      return ESP_FAIL;

    const int8_t* qlogits = output->data.int8;
    logits.resize(10);
    std::transform(qlogits, qlogits+10, logits.begin(), [output](int8_t v){
        return (v - output->params.zero_point) * output->params.scale;
      });

    return ESP_OK;
  }

  esp_err_t Model::UpdateModel(const char* modelData, std::size_t dataSize)
  {
    const tflite::Model* model = ParseTfLiteModel(reinterpret_cast<const uint8_t*>(modelData));
    if (!model)
      return ESP_FAIL;

    resolver = CreateOpResolver();
    if (!resolver)
      return ESP_FAIL;

    interpreter = CreateInterpreter(model, *resolver, TensorArena, TensorArenaSize);
    if (!interpreter)
      return ESP_FAIL;

    if (interpreter->Invoke() != kTfLiteOk)
    {
      ESP_LOGE(TAG, "Failed to run inference on new model.");
      return ESP_FAIL;
    }

    esp_err_t err = modelLoader.UpdateModel(reinterpret_cast<const uint8_t*>(modelData), 0, dataSize);
    ESP_RETURN_ON_ERROR(err, TAG, "Failed to write model");

    Deinit();
    err = Init();
    ESP_RETURN_ON_ERROR(err, TAG, "Failed to initialize new model");


    ESP_LOGI(TAG, "Model sucessfully updated.");
    return ESP_OK;
  }
}
