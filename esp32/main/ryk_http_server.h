#pragma once

#include <esp_err.h>

namespace Ryk
{
  class Model;

  esp_err_t StartHttpServer(Model& model);
}
