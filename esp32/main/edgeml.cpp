#include <stdio.h>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"

extern "C"
void app_main(void)
{
  printf("Hello, world!");
  MicroPrintf("Hello, TFLite Micro!");
}
