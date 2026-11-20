#include "ryk_model.h"


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

    const tflite::Model* model = tflite::GetModel(modelData);
    if (model == nullptr)
    {
      MicroPrintf("Failed to parse model");
      return ESP_FAIL;
    }

    if (model->version() != TFLITE_SCHEMA_VERSION)
    {
      MicroPrintf("Unsupportd model version %i, required %i", (int)model->version(), (int)TFLITE_SCHEMA_VERSION);
      return ESP_FAIL;
    }

    resolver = std::make_unique<tflite::MicroMutableOpResolver<10>>();
    resolver->AddConv2D();
    resolver->AddAveragePool2D();
    resolver->AddFullyConnected();
    resolver->AddReshape();
    resolver->AddPad();
    resolver->AddDepthwiseConv2D();
    resolver->AddTranspose();
    resolver->AddLogistic();

    interpreter = std::make_unique<tflite::MicroInterpreter>(
      model,
      *resolver,
      TensorArena,
      TensorArenaSize
    );

    TfLiteStatus status = interpreter->AllocateTensors();
    if (status != kTfLiteOk)
    {
      MicroPrintf("Failed to allocate tensors");
      return ESP_FAIL;
    }

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

  esp_err_t Model::UpdateModel(const std::vector<uint8_t>& modelData)
  {
    return ESP_OK;
  }
}
