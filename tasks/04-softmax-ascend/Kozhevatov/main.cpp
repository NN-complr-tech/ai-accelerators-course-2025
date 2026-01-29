/**
 * @file main.cpp
 * Softmax implementation reading from file
 */
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>

#include "data_utils.h"  // Предполагаем, что здесь есть WriteFile

#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
extern void softmax_custom_do(uint32_t blockDim, void* stream, uint8_t* x,
                              uint8_t* y);
#else
#include "tikicpulib.h"
extern "C" __global__ __aicore__ void softmax_custom(GM_ADDR x, GM_ADDR y);
#endif

using namespace std;

// Вспомогательная функция для чтения бинарного файла
bool ReadBinFile(const string& filename, void* data, size_t size) {
  ifstream file(filename, ios::binary);
  if (!file.is_open()) {
    cerr << "[ERROR] Failed to open file: " << filename << endl;
    return false;
  }
  file.read((char*)data, size);
  if (!file) {
    cerr << "[ERROR] Failed to read complete data from: " << filename << endl;
    return false;
  }
  return true;
}

int32_t main(int32_t argc, char* argv[]) {
  uint32_t blockDim = 8;
  size_t vectorSize = 8 * 2048;
  size_t byteSize = vectorSize * sizeof(float);

  // Выделяем память на хосте
  float* hostInput = new float[vectorSize];
  float* hostOutput = new float[vectorSize];

  // 1. ЧИТАЕМ ВХОДНЫЕ ДАННЫЕ, СГЕНЕРИРОВАННЫЕ PYTHON
  // Путь должен совпадать с тем, куда пишет gen_data.py
  if (!ReadBinFile("./input/softmax_input.bin", hostInput, byteSize)) {
    delete[] hostInput;
    delete[] hostOutput;
    return -1;
  }

#ifdef ASCENDC_CPU_DEBUG
  cout << "Running in CPU Debug mode..." << endl;
  uint8_t* x = (uint8_t*)AscendC::GmAlloc(byteSize);
  uint8_t* y = (uint8_t*)AscendC::GmAlloc(byteSize);

  memcpy(x, hostInput, byteSize);

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(softmax_custom, blockDim, x, y);

  memcpy(hostOutput, y, byteSize);
  AscendC::GmFree((void*)x);
  AscendC::GmFree((void*)y);
#else
  cout << "Running in NPU/SIM mode..." << endl;

  // Инициализация ACL
  aclInit(nullptr);
  aclrtSetDevice(0);
  aclrtStream stream = nullptr;
  aclrtCreateStream(&stream);

  uint8_t *xDevice = nullptr, *yDevice = nullptr;
  uint8_t *xHost = nullptr, *yHost = nullptr;

  // Pinned memory on Host
  aclrtMallocHost((void**)(&xHost), byteSize);
  aclrtMallocHost((void**)(&yHost), byteSize);

  // Memory on Device
  aclrtMalloc((void**)&xDevice, byteSize, ACL_MEM_MALLOC_HUGE_FIRST);
  aclrtMalloc((void**)&yDevice, byteSize, ACL_MEM_MALLOC_HUGE_FIRST);

  // Копируем прочитанные данные в pinned memory
  memcpy(xHost, hostInput, byteSize);

  // Host -> Device
  aclrtMemcpy(xDevice, byteSize, xHost, byteSize, ACL_MEMCPY_HOST_TO_DEVICE);

  // Запуск ядра
  softmax_custom_do(blockDim, stream, xDevice, yDevice);
  aclrtSynchronizeStream(stream);

  // Device -> Host
  aclrtMemcpy(yHost, byteSize, yDevice, byteSize, ACL_MEMCPY_DEVICE_TO_HOST);

  // Сохраняем результат в обычный буфер
  memcpy(hostOutput, yHost, byteSize);

  // Очистка
  aclrtFree(xDevice);
  aclrtFree(yDevice);
  aclrtFreeHost(xHost);
  aclrtFreeHost(yHost);
  aclrtDestroyStream(stream);
  aclrtResetDevice(0);
  aclFinalize();
#endif

  // 2. ЗАПИСЫВАЕМ РЕЗУЛЬТАТ ДЛЯ ПРОВЕРКИ PYTHON-СКРИПТОМ
  // Имя файла должно совпадать с аргументом в run.sh (output_z.bin)
  WriteFile("./output/output_z.bin", (uint8_t*)hostOutput, byteSize);
  cout << "[INFO] Results saved to ./output/output_z.bin" << endl;

  delete[] hostInput;
  delete[] hostOutput;
  return 0;
}