#pragma once

#include <onnxruntime_lite_custom_op.h>

namespace Ryk
{
  void Relu(const Ort::Custom::Tensor<float>& X, Ort::Custom::Tensor<float>& Y);

  class OpDomain : public Ort::CustomOpDomain
  {
  public:
    OpDomain()
      : Ort::CustomOpDomain("ryk")
      , RykRelu(Ort::Custom::CreateLiteCustomOp("RykRelu", "CPUExecutionProvider", Relu))
    {
      // These callback define indeces of input tensors can be reused for corresponding output tensors
      RykRelu->GetMayInplace = [](int** inputIndeces, int** outputIndeces) -> size_t
        {
          *inputIndeces = new int[1]{0};
          *outputIndeces = new int[1]{0};
          return 1;
        };
      RykRelu->ReleaseMayInplace = [](int* inputIndeces, int* outputIndeces)
        {
          delete[] inputIndeces;
          delete[] outputIndeces;
        };

      Add(RykRelu.get());
    }

  private:
    std::unique_ptr<Ort::Custom::OrtLiteCustomOp> RykRelu;
    std::array<int, 1> MayInplace = {0};
  };
}
