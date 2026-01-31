/**
 * @file main.cpp
 * Main program для MatMul+LeakyReLU + Softmax
 *
 * Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "config.h"
#include "data_utils.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tiling/platform/platform_ascendc.h"

#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
#include "aclrtlaunch_matmul_leakyrelu_custom.h"
#include "aclrtlaunch_softmax_custom.h"
extern void softmax_custom_do(uint32_t blockDim, void* stream, uint8_t* x,
                              uint8_t* z);
#else
#include "tikicpulib.h"
extern "C" void matmul_leakyrelu_custom(uint8_t*, uint8_t*, uint8_t*, uint8_t*,
                                        uint8_t*, uint8_t*);
extern "C" __global__ __aicore__ void softmax_custom(GM_ADDR x, GM_ADDR z);
#endif

extern void GenerateTiling(const char* socVersion, uint8_t* tilingBuf);

int32_t main(int32_t argc, char* argv[]) {
  const char* socVersion = SOC_VERSION;
  auto ascendcPlatform =
      platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);

  // Размеры данных из конфига (как в примере, но вычисляемые)
  size_t aFileSize = MATRIX_SIZE * MATRIX_SIZE * sizeof(int16_t);  // half для A
  size_t bFileSize = MATRIX_SIZE * MATRIX_SIZE * sizeof(int16_t);  // half для B
  size_t biasFileSize = MATRIX_SIZE * sizeof(float);  // float для bias
  size_t matmulOutputSize =
      MATRIX_SIZE * MATRIX_SIZE * sizeof(float);  // MatMul результат
  size_t softmaxOutputSize =
      MATRIX_SIZE * MATRIX_SIZE * sizeof(float);  // Softmax результат

  size_t tilingFileSize = sizeof(TCubeTiling);
  size_t userWorkspaceSize = 0;
  size_t systemWorkspaceSize =
      static_cast<size_t>(ascendcPlatform->GetLibApiWorkSpaceSize());
  size_t workspaceSize = userWorkspaceSize + systemWorkspaceSize;

  uint8_t* tilingBuf = (uint8_t*)malloc(tilingFileSize);
  GenerateTiling(socVersion, tilingBuf);

#ifdef CUSTOM_ASCEND310P
  uint32_t matmulBlockDim = MATMUL_CORE_NUM;
#else
  uint32_t matmulBlockDim = 1;
#endif

  uint32_t softmaxBlockDim = USE_CORE_NUM;

#ifdef ASCENDC_CPU_DEBUG
  // Выделение памяти (как в примере)
  uint8_t* a = (uint8_t*)AscendC::GmAlloc(aFileSize);
  uint8_t* b = (uint8_t*)AscendC::GmAlloc(bFileSize);
  uint8_t* bias = (uint8_t*)AscendC::GmAlloc(biasFileSize);
  uint8_t* matmul_output = (uint8_t*)AscendC::GmAlloc(matmulOutputSize);
  uint8_t* softmax_output = (uint8_t*)AscendC::GmAlloc(softmaxOutputSize);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingFileSize);
  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);

  // Чтение входных данных (как в примере)
  ReadFile("./input/x1_gm.bin", aFileSize, a, aFileSize);
  ReadFile("./input/x2_gm.bin", bFileSize, b, bFileSize);
  ReadFile("./input/bias.bin", biasFileSize, bias, biasFileSize);
  memcpy_s(tiling, tilingFileSize, tilingBuf, tilingFileSize);

  // Запуск MatMul (как в примере)
  ICPU_RUN_KF(matmul_leakyrelu_custom, matmulBlockDim, a, b, bias,
              matmul_output, workspace, tiling);

  // Переключение режима для Softmax (из условия задания)
  AscendC::SetKernelMode(KernelMode::AIV_MODE);

  // Запуск Softmax (добавлено)
  ICPU_RUN_KF(softmax_custom, softmaxBlockDim, matmul_output, softmax_output);

  // Запись результатов (как в примере, но два файла)
  WriteFile("./output/matmul_output.bin", matmul_output, matmulOutputSize);
  WriteFile("./output/softmax_output.bin", softmax_output, softmaxOutputSize);

  // Освобождение памяти (как в примере)
  AscendC::GmFree((void*)a);
  AscendC::GmFree((void*)b);
  AscendC::GmFree((void*)bias);
  AscendC::GmFree((void*)matmul_output);
  AscendC::GmFree((void*)softmax_output);
  AscendC::GmFree((void*)tiling);
  AscendC::GmFree((void*)workspace);
#else
  // Режим работы на устройстве (как в примере)
  CHECK_ACL(aclInit(nullptr));
  int32_t deviceId = 0;
  CHECK_ACL(aclrtSetDevice(deviceId));
  aclrtStream stream = nullptr;
  CHECK_ACL(aclrtCreateStream(&stream));

  // A matrix (как в примере)
  uint8_t* inputAHost;
  uint8_t* inputADevice;
  CHECK_ACL(aclrtMallocHost((void**)(&inputAHost), aFileSize));
  CHECK_ACL(
      aclrtMalloc((void**)&inputADevice, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
  ReadFile("./input/x1_gm.bin", aFileSize, inputAHost, aFileSize);
  CHECK_ACL(aclrtMemcpy(inputADevice, aFileSize, inputAHost, aFileSize,
                        ACL_MEMCPY_HOST_TO_DEVICE));

  // B matrix (как в примере)
  uint8_t* inputBHost;
  uint8_t* inputBDevice;
  CHECK_ACL(aclrtMallocHost((void**)(&inputBHost), bFileSize));
  CHECK_ACL(
      aclrtMalloc((void**)&inputBDevice, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
  ReadFile("./input/x2_gm.bin", bFileSize, inputBHost, bFileSize);
  CHECK_ACL(aclrtMemcpy(inputBDevice, bFileSize, inputBHost, bFileSize,
                        ACL_MEMCPY_HOST_TO_DEVICE));

  // Bias (как в примере)
  uint8_t* inputBiasHost;
  uint8_t* inputBiasDevice;
  CHECK_ACL(aclrtMallocHost((void**)(&inputBiasHost), biasFileSize));
  CHECK_ACL(aclrtMalloc((void**)&inputBiasDevice, biasFileSize,
                        ACL_MEM_MALLOC_HUGE_FIRST));
  ReadFile("./input/bias.bin", biasFileSize, inputBiasHost, biasFileSize);
  CHECK_ACL(aclrtMemcpy(inputBiasDevice, biasFileSize, inputBiasHost,
                        biasFileSize, ACL_MEMCPY_HOST_TO_DEVICE));

  // MatMul output (как в примере)
  uint8_t* matmulOutputHost;
  uint8_t* matmulOutputDevice;
  CHECK_ACL(aclrtMallocHost((void**)(&matmulOutputHost), matmulOutputSize));
  CHECK_ACL(aclrtMalloc((void**)&matmulOutputDevice, matmulOutputSize,
                        ACL_MEM_MALLOC_HUGE_FIRST));

  // Softmax output (добавлен)
  uint8_t* softmaxOutputHost;
  uint8_t* softmaxOutputDevice;
  CHECK_ACL(aclrtMallocHost((void**)(&softmaxOutputHost), softmaxOutputSize));
  CHECK_ACL(aclrtMalloc((void**)&softmaxOutputDevice, softmaxOutputSize,
                        ACL_MEM_MALLOC_HUGE_FIRST));

  // Tiling (как в примере)
  uint8_t* tilingHost;
  uint8_t* tilingDevice;
  CHECK_ACL(aclrtMallocHost((void**)(&tilingHost), tilingFileSize));
  CHECK_ACL(aclrtMalloc((void**)&tilingDevice, tilingFileSize,
                        ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMemcpy(tilingHost, tilingFileSize, tilingBuf, tilingFileSize,
                        ACL_MEMCPY_HOST_TO_HOST));
  CHECK_ACL(aclrtMemcpy(tilingDevice, tilingFileSize, tilingHost,
                        tilingFileSize, ACL_MEMCPY_HOST_TO_DEVICE));

  // Workspace (как в примере)
  uint8_t* workspaceDevice;
  CHECK_ACL(aclrtMalloc((void**)&workspaceDevice, workspaceSize,
                        ACL_MEM_MALLOC_HUGE_FIRST));

  // Запуск MatMul ядра (как в примере)
  ACLRT_LAUNCH_KERNEL(matmul_leakyrelu_custom)
  (matmulBlockDim, stream, inputADevice, inputBDevice, inputBiasDevice,
   matmulOutputDevice, workspaceDevice, tilingDevice);

  CHECK_ACL(aclrtSynchronizeStream(stream));

  // Запуск Softmax ядра (добавлено)
  softmax_custom_do(softmaxBlockDim, stream, matmulOutputDevice,
                    softmaxOutputDevice);

  CHECK_ACL(aclrtSynchronizeStream(stream));

  // Копирование результатов MatMul (как в примере)
  CHECK_ACL(aclrtMemcpy(matmulOutputHost, matmulOutputSize, matmulOutputDevice,
                        matmulOutputSize, ACL_MEMCPY_DEVICE_TO_HOST));

  // Копирование результатов Softmax (добавлено)
  CHECK_ACL(aclrtMemcpy(softmaxOutputHost, softmaxOutputSize,
                        softmaxOutputDevice, softmaxOutputSize,
                        ACL_MEMCPY_DEVICE_TO_HOST));

  // Запись файлов (как в примере, но два файла)
  WriteFile("./output/matmul_output.bin", matmulOutputHost, matmulOutputSize);
  WriteFile("./output/softmax_output.bin", softmaxOutputHost,
            softmaxOutputSize);

  // Освобождение памяти (как в примере)
  CHECK_ACL(aclrtFree(inputADevice));
  CHECK_ACL(aclrtFreeHost(inputAHost));
  CHECK_ACL(aclrtFree(inputBDevice));
  CHECK_ACL(aclrtFreeHost(inputBHost));
  CHECK_ACL(aclrtFree(inputBiasDevice));
  CHECK_ACL(aclrtFreeHost(inputBiasHost));
  CHECK_ACL(aclrtFree(matmulOutputDevice));
  CHECK_ACL(aclrtFreeHost(matmulOutputHost));
  CHECK_ACL(aclrtFree(softmaxOutputDevice));
  CHECK_ACL(aclrtFreeHost(softmaxOutputHost));
  CHECK_ACL(aclrtFree(tilingDevice));
  CHECK_ACL(aclrtFreeHost(tilingHost));
  CHECK_ACL(aclrtFree(workspaceDevice));

  CHECK_ACL(aclrtDestroyStream(stream));
  CHECK_ACL(aclrtResetDevice(deviceId));
  CHECK_ACL(aclFinalize());
#endif

  free(tilingBuf);

  // вывод (дабавлен)
  std::cout << "✓ Program completed successfully" << std::endl;
  std::cout << "  Matrix size: " << MATRIX_SIZE << "x" << MATRIX_SIZE
            << std::endl;
  std::cout << "  Softmax cores: " << USE_CORE_NUM << std::endl;
  std::cout << "  MatMul cores: " << MATMUL_CORE_NUM << std::endl;

  return 0;
}