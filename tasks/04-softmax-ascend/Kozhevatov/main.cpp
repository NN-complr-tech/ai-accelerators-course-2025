/**
 * @file main.cpp
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */

#include <chrono>
#include <iostream>

#include "data_utils.h"

#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
extern void softmax_custom_do(uint32_t blockDim, void *stream, uint8_t *x,
                              uint8_t *z);
#else
#include "tikicpulib.h"
extern "C" __global__ __aicore__ void softmax_custom(GM_ADDR x, GM_ADDR z);
#endif

constexpr long BLOCK_DIM = 16;
constexpr long SIZE = 64;

int32_t main(int32_t argc, char *argv[]) {
  uint32_t blockDim = BLOCK_DIM;
  size_t inputByteSize = SIZE * SIZE * sizeof(float);
  size_t outputByteSize = SIZE * SIZE * sizeof(float);

  std::cout << "=== Softmax Custom Kernel ===" << std::endl;
  std::cout << "Matrix: " << SIZE << "x" << SIZE << " (" << SIZE * SIZE
            << " elements)" << std::endl;

  auto totalStart = std::chrono::high_resolution_clock::now();

#ifdef ASCENDC_CPU_DEBUG
  std::cout << "Mode: CPU (DEBUG)" << std::endl;

  // CPU режим
  uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputByteSize);
  uint8_t *z = (uint8_t *)AscendC::GmAlloc(outputByteSize);

  ReadFile("./input/input_matrix.bin", inputByteSize, x, inputByteSize);

  AscendC::SetKernelMode(KernelMode::AIV_MODE);

  auto kernelStart = std::chrono::high_resolution_clock::now();
  ICPU_RUN_KF(softmax_custom, blockDim, x, z);
  auto kernelEnd = std::chrono::high_resolution_clock::now();

  WriteFile("./output/output_z.bin", z, outputByteSize);

  AscendC::GmFree((void *)x);
  AscendC::GmFree((void *)z);

  auto kernelDuration = std::chrono::duration_cast<std::chrono::milliseconds>(
      kernelEnd - kernelStart);
  std::cout << "Kernel time: " << kernelDuration.count() << " ms" << std::endl;

#else
  std::cout << "Mode: NPU/SIM" << std::endl;

  // NPU/SIM режим
  CHECK_ACL(aclInit(nullptr));
  int32_t deviceId = 0;
  CHECK_ACL(aclrtSetDevice(deviceId));
  aclrtStream stream = nullptr;
  CHECK_ACL(aclrtCreateStream(&stream));

  uint8_t *xHost, *zHost;
  uint8_t *xDevice, *zDevice;

  CHECK_ACL(aclrtMallocHost((void **)(&xHost), inputByteSize));
  CHECK_ACL(aclrtMallocHost((void **)(&zHost), outputByteSize));
  CHECK_ACL(
      aclrtMalloc((void **)&xDevice, inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMalloc((void **)&zDevice, outputByteSize,
                        ACL_MEM_MALLOC_HUGE_FIRST));

  ReadFile("./input/input_matrix.bin", inputByteSize, xHost, inputByteSize);

  CHECK_ACL(aclrtMemcpy(xDevice, inputByteSize, xHost, inputByteSize,
                        ACL_MEMCPY_HOST_TO_DEVICE));

  auto kernelStart = std::chrono::high_resolution_clock::now();
  softmax_custom_do(blockDim, stream, xDevice, zDevice);
  CHECK_ACL(aclrtSynchronizeStream(stream));
  auto kernelEnd = std::chrono::high_resolution_clock::now();

  CHECK_ACL(aclrtMemcpy(zHost, outputByteSize, zDevice, outputByteSize,
                        ACL_MEMCPY_DEVICE_TO_HOST));
  WriteFile("./output/output_z.bin", zHost, outputByteSize);

  CHECK_ACL(aclrtFree(xDevice));
  CHECK_ACL(aclrtFree(zDevice));
  CHECK_ACL(aclrtFreeHost(xHost));
  CHECK_ACL(aclrtFreeHost(zHost));

  CHECK_ACL(aclrtDestroyStream(stream));
  CHECK_ACL(aclrtResetDevice(deviceId));
  CHECK_ACL(aclFinalize());

  auto kernelDuration = std::chrono::duration_cast<std::chrono::milliseconds>(
      kernelEnd - kernelStart);
  std::cout << "Kernel time: " << kernelDuration.count() << " ms" << std::endl;
#endif

  auto totalEnd = std::chrono::high_resolution_clock::now();
  auto totalDuration = std::chrono::duration_cast<std::chrono::milliseconds>(
      totalEnd - totalStart);

  std::cout << "Total time: " << totalDuration.count() << " ms" << std::endl;
  std::cout << "===============================" << std::endl;

  return 0;
}