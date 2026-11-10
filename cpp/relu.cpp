#include "relu.h"

#include <algorithm>


void Ryk::Relu(const Ort::Custom::Tensor<float>& X, Ort::Custom::Tensor<float>& Y)
{
  auto xData = X.Data();
  auto yData = Y.Allocate(X.Shape());

  for (std::size_t i = 0; i < X.NumberOfElement(); ++i)
  {
    yData[i] = std::max<float>(0.0f, xData[i]);
  }
}
