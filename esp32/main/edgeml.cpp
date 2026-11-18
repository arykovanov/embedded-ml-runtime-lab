#include <stdio.h>
#include <algorithm>
#include <numeric>
#include <cinttypes>

#include <esp_heap_caps.h>
#include <esp_timer.h>
#include <nvs_flash.h>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

#include <tensor_utils.h>

#include "load_model.h"

extern const uint8_t sample_images_start[]
    asm("_binary_sample_images_bin_start");
extern const uint8_t sample_images_end[]
    asm("_binary_sample_images_bin_end");

static const size_t TensorArenaSize = 19 * 1024;
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

void PrintMemoryStats(const char* title)
{
  MicroPrintf("ESP32 memory %s:", title ? title : "");
  MicroPrintf("\tFree RAM: %u", heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
  MicroPrintf("\tFree PSRAM: %u", heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
}

struct StatsMeasure
{
  uint64_t Us = 0;
  void Start()
  {
    Us = esp_timer_get_time();
  }

  void Stop()
  {
    Us = esp_timer_get_time() - Us;
  }
};

struct PerformanceStat
{
  StatsMeasure Initialization;
  std::vector<StatsMeasure> Inferences;
};


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

extern "C"
void app_main(void)
{
  PerformanceStat perfStat;
  NVSHolder nvs;

  esp_err_t err = nvs.Init();
  if (err != ESP_OK)
  {
    MicroPrintf("Failed to initialize NVS partition: %u - %s", err, esp_err_to_name(err));
    return;
  }

  Ryk::ModelStorage modelLoader;
  const uint8_t* modelData = nullptr;
  err = modelLoader.GetModelData(&modelData);
  if (err != ESP_OK)
  {
    MicroPrintf("Failed to load model: %u - %s", err, esp_err_to_name(err));
    return;
  }

  MicroPrintf("Starting Inference");
  PrintMemoryStats("initial");

  const Ryk::SavedTensor sampleImages = Ryk::LoadSampleBatch(sample_images_start, sample_images_end);
  MicroPrintf("Sample image shape: %s", concat(sampleImages.Shape).c_str());

  PrintMemoryStats("after loading sample images");

  perfStat.Initialization.Start();

  resolver.AddConv2D();
  resolver.AddAveragePool2D();
  resolver.AddFullyConnected();
  resolver.AddReshape();
  resolver.AddPad();
  resolver.AddDepthwiseConv2D();
  resolver.AddTranspose();
  resolver.AddLogistic();

  const tflite::Model* model = tflite::GetModel(modelData);
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

  perfStat.Initialization.Stop();

  PrintMemoryStats("after allocating tensors");

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

  PrintMemoryStats("before inference");

  perfStat.Inferences.resize(sampleImages.Shape[0]);

  for (size_t i = 0; i < sampleImages.Shape[0]; ++i)
  {
    for (size_t qpos = 0, dataPos = i * input->bytes; qpos < input->bytes; ++qpos)
    {
      const int8_t q = static_cast<int8_t>(std::clamp(std::round(sampleImages.Data[dataPos + qpos] / scale + zeroPoint), -128.0f, 127.0f));
      input->data.int8[qpos] = q;
    }

    perfStat.Inferences[i].Start();

    if (interpreter.Invoke() != kTfLiteOk)
    {
      MicroPrintf("interpreter invoe retuendt error");
      return;
    }

    perfStat.Inferences[i].Stop();

    const TfLiteTensor* output = interpreter.output(0);
    const int8_t* qlogits = output->data.int8;
    std::vector<float> logits(10);
    std::transform(qlogits, qlogits+10, logits.begin(), [output](int8_t v){
      return (v - output->params.zero_point) * output->params.scale;
    });

    MicroPrintf("logits #%zu: %s", i, concat(logits).c_str());
  }

  PrintMemoryStats("after inference");

  MicroPrintf("Inference statistics:");
  MicroPrintf("\tArena used bytes: %zu", interpreter.arena_used_bytes());
  MicroPrintf("\tInitialization(us): %" PRIu64, perfStat.Initialization.Us);
  MicroPrintf("\tFirst inference time(us): %" PRIu64, perfStat.Inferences[0].Us);

  uint64_t totalUs = 0;
  for (const auto& inference : perfStat.Inferences)
    totalUs += inference.Us;

  MicroPrintf("\tTotal time(us): %" PRIu64, totalUs);

  auto p95 = perfStat.Inferences;
  std::sort(p95.begin(), p95.end(), [](auto a, auto b) {return a.Us < b.Us;});

  MicroPrintf("\tMin time(us): %" PRIu64, p95[0].Us);
  MicroPrintf("\tMax time(us): %" PRIu64, p95.back().Us);

  if (p95.size() % 2 == 0)
  {
    const auto border = (p95.size() - 1) / 2;
    MicroPrintf("\tMedian time(us): %" PRIu64, (p95[border].Us + p95[border+1].Us)/2);
  }
  else
  {
    const auto border = p95.size() / 2;
    MicroPrintf("\tMedian time(us): %" PRIu64, p95[border].Us);
  }

  MicroPrintf("\tP95 time(us): %" PRIu64, p95[p95.size() * 0.95].Us);
  MicroPrintf("\tP90 time(us): %" PRIu64, p95[p95.size() * 0.9].Us);

  MicroPrintf("Done");
}
