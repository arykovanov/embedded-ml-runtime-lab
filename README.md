# AI Engineer experiments

This repository and its linked Github project document a 20-day challenge to
learn end-to-end process of developing a simple Neural Network and runing it on an ESP32.

## Phase 1 - Develop a CNN

In this phase, I'll train a LeNet-like neural network on Fashion MNIST dataset using
PyTorch and compare its inference with those from ONNX runtime.

Then I'll develop a simple C++ program to run same model.
C++ program will be linked against ONNC Runtime

I'll compare the inference output in a Jupyter Notebook.

## Phase 2 - Explore inference runtime internals

In this phase, I'll investigate how ONNX Runtime optimizes a model's graph,
selects operator implementations, manages tensor memory, and executes inference.
I'll implement simple reference operators and a custom ONNX Runtime operator,
then profile CPU inference to identify performance bottlenecks.

The purpose is to understand what happens behind the inference API and gain
practical experience debugging correctness and performance issues in a runtime.

## Pase 3 - Embedded inference on ESP32

In this phase we will quantize model, optimize, convert to TFLite Micro, check inference and run model on the ESP32 board.

## Phase 4 - Hot model replacement and optimizations

In this phase, I'll implement model replacement without rebooting the ESP32,
including compatibility checks and handling loading failures. I'll also build
a small backend-selection and CPU-fallback prototype, investigate a measured
performance bottleneck, and validate any improvements against reference outputs.

I'll document the architecture, benchmarks, and limitations, then validate the
complete pipeline and practise explaining it in a mock technical interview.
The purpose is to develop practical runtime engineering skills and demonstrate
how I diagnose problems, evaluate trade-offs, and communicate technical results.
