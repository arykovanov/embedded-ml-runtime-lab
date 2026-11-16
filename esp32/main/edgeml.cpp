#include <stdio.h>
#include <algorithm>
#include <numeric>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

#include <tensor_utils.h>

extern const uint8_t model_start[]
    asm("_binary_lenet_quantized_tflite_start");
extern const uint8_t model_end[]
    asm("_binary_lenet_quantized_tflite_end");

extern const uint8_t sample_images_start[]
    asm("_binary_sample_images_bin_start");
extern const uint8_t sample_images_end[]
    asm("_binary_sample_images_bin_end");

static const size_t TensorArenaSize = 260 * 1024;
static uint8_t TensorArena[TensorArenaSize] = {};
static tflite::MicroMutableOpResolver<10> resolver;

template <typename Iter>
std::string concat(const Iter& begin, const Iter& end)
{
  if (begin == end)
    return {};

  return std::accumulate(std::next(begin), end, std::to_string(*begin),
    [](const std::string& s, auto num)
    {
      return s + ", " + std::to_string(num);
    }
  );
}

template <typename T>
std::string concat(const std::vector<T>& v)
{
  return concat(v.begin(), v.end());
}

extern "C"
void app_main(void)
{
  MicroPrintf("Starting Inference");

  resolver.AddConv2D();
  resolver.AddAveragePool2D();
  resolver.AddFullyConnected();
  resolver.AddReshape();
  resolver.AddPad();
  resolver.AddDepthwiseConv2D();
  resolver.AddTranspose();
  resolver.AddLogistic();

  const Ryk::SavedTensor sampleImages = Ryk::LoadSampleBatch(sample_images_start, sample_images_end);
  MicroPrintf("Sample image shape: %s", concat(sampleImages.Shape).c_str());

  const tflite::Model* model = tflite::GetModel(model_start);
  if (model == nullptr)
  {
    MicroPrintf("Failed to load model");
    return;
  }

  if (model->version() != TFLITE_SCHEMA_VERSION)
  {
    MicroPrintf("Invalid model version %u, expected version %u", model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  tflite::MicroInterpreter interpreter(
    model,
    resolver,
    TensorArena,
    TensorArenaSize
  );

  TfLiteStatus status = interpreter.AllocateTensors();
  if (status != kTfLiteOk)
  {
    MicroPrintf("Failed to allocate tensors");
    return;
  }

  TfLiteTensor* input = interpreter.input(0);
  MicroPrintf("Input tensor:");
  MicroPrintf("\ttype: %d", input->type);
  MicroPrintf("\t#dims: %d", input->dims->size);
  for (unsigned i = 0; i < input->dims->size; ++i)
  {
    MicroPrintf("\t\tdim[%i]: %u", i, input->dims->data[i]);
  }
  MicroPrintf("\tbytes: %u", input->bytes);

  if (!std::equal(std::next(sampleImages.Shape.begin()), sampleImages.Shape.end(), input->dims->data + 1))
  {
    MicroPrintf("Input Tensor and has different shape than image");
    return;
  }

  MicroPrintf("\t Quantization: zero=%i, scale=*f", input->params.zero_point, input->params.scale);
  const TfLiteTensor* output = interpreter.output(0);
  MicroPrintf("Output tensor:");
  MicroPrintf("\ttype: %d", output->type);
  MicroPrintf("\t#dims: %d", output->dims->size);
  for (unsigned i = 0; i < output->dims->size; ++i)
  {
    MicroPrintf("\t\tdim[%i]: %u", i, output->dims->data[i]);
  }
  MicroPrintf("\tbytes: %u", output->bytes);

  const int32_t zeroPoint = input->params.zero_point;
  const float scale = input->params.scale;

  for (size_t i = 0; i < sampleImages.Shape[0]; ++i)
  {
    MicroPrintf("Processing image: %i", i);

    for (size_t qpos = 0, dataPos = i * input->bytes; qpos < input->bytes; ++qpos)
    {
      const int8_t q = static_cast<int8_t>(std::clamp(std::round(sampleImages.Data[dataPos + qpos] / scale + zeroPoint), -128.0f, 127.0f));
      input->data.int8[qpos] = q;
    }

    if (interpreter.Invoke() != kTfLiteOk)
    {
      MicroPrintf("interpreter invoe retuendt error");
      return;
    }
    const TfLiteTensor* output = interpreter.output(0);
    const int8_t* qlogits = output->data.int8;
    std::vector<float> logits(10);
    std::transform(qlogits, qlogits+10, logits.begin(), [output](int8_t v){
      return (v - output->params.zero_point) * output->params.scale;
    });

    MicroPrintf("logits: %s", concat(logits).c_str());
  }

  MicroPrintf("Done");
}
