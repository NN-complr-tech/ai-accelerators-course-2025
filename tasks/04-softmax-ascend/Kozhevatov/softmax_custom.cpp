/**
 * @file softmax_custom.cpp
 * Стабильная версия Softmax для CPU и Симулятора
 */

#include "kernel_operator.h"

using namespace AscendC;

// Константы
constexpr int32_t TOTAL_LENGTH = 8 * 2048;
constexpr int32_t USE_CORE_NUM = 8;
constexpr int32_t BLOCK_LENGTH = TOTAL_LENGTH / USE_CORE_NUM;
constexpr int32_t TILE_NUM = 8;
constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t TILE_LENGTH = BLOCK_LENGTH / TILE_NUM / BUFFER_NUM;

class KernelSoftmax {
 public:
  __aicore__ inline KernelSoftmax() {}

  __aicore__ inline void Init(GM_ADDR x, GM_ADDR y) {
    // Настройка смещения для каждого ядра (блока)
    xGm.SetGlobalBuffer((__gm__ float*)x + BLOCK_LENGTH * GetBlockIdx(),
                        BLOCK_LENGTH);
    yGm.SetGlobalBuffer((__gm__ float*)y + BLOCK_LENGTH * GetBlockIdx(),
                        BLOCK_LENGTH);

    // Инициализация очередей
    pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_LENGTH * sizeof(float));
    pipe.InitBuffer(expQueue, BUFFER_NUM, TILE_LENGTH * sizeof(float));
    pipe.InitBuffer(outQueueY, BUFFER_NUM, TILE_LENGTH * sizeof(float));
    pipe.InitBuffer(tmpQueue, 1, TILE_LENGTH * sizeof(float));
  }

  __aicore__ inline void Process() {
    // 1. Находим максимум в блоке (для стабильности exp)
    float blockMax = FindBlockMax();

    // 2. Считаем сумму экспонент и сохраняем промежуточные exp(x - max) в
    // Global Memory
    float totalSum = ComputeExpSumAndStore(blockMax);

    // 3. Финальная нормализация: y = exp / totalSum
    Normalize(totalSum);
  }

 private:
  // Поиск максимума во всем блоке (2048 элементов)
  __aicore__ inline float FindBlockMax() {
    float blockMax = -1e38f;
    for (int32_t i = 0; i < TILE_NUM * BUFFER_NUM; i++) {
      LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
      DataCopy(xLocal, xGm[i * TILE_LENGTH], TILE_LENGTH);

      // Поэлементный поиск максимума в тайле
      for (int32_t j = 0; j < TILE_LENGTH; j++) {
        float val = xLocal.GetValue(j);
        if (val > blockMax) blockMax = val;
      }
      inQueueX.FreeTensor(xLocal);
    }
    return blockMax;
  }

  // Вычисление экспонент и их суммы
  __aicore__ inline float ComputeExpSumAndStore(float blockMax) {
    float totalSum = 0.0f;

    for (int32_t i = 0; i < TILE_NUM * BUFFER_NUM; i++) {
      LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
      LocalTensor<float> expLocal = expQueue.AllocTensor<float>();
      LocalTensor<float> workLocal = outQueueY.AllocTensor<float>();

      DataCopy(xLocal, xGm[i * TILE_LENGTH], TILE_LENGTH);

      // Стабилизация: x - max
      // Создаем временный тензор со значением max для вычитания
      for (int32_t j = 0; j < TILE_LENGTH; j++) workLocal.SetValue(j, blockMax);
      Sub(expLocal, xLocal, workLocal, TILE_LENGTH);

      // exp(x - max)
      Exp(expLocal, expLocal, TILE_LENGTH);

      // Считаем сумму текущего тайла
      for (int32_t j = 0; j < TILE_LENGTH; j++) {
        totalSum += expLocal.GetValue(j);
      }

      // Сохраняем результат экспоненты временно в выходной буфер Global Memory
      DataCopy(yGm[i * TILE_LENGTH], expLocal, TILE_LENGTH);

      outQueueY.FreeTensor(workLocal);
      expQueue.FreeTensor(expLocal);
      inQueueX.FreeTensor(xLocal);
    }
    return totalSum;
  }

  // Деление на общую сумму (Нормализация)
  __aicore__ inline void Normalize(float totalSum) {
    // Избегаем деления на ноль
    float reciprocal = (totalSum > 0.0f) ? (1.0f / totalSum) : 0.0f;

    for (int32_t i = 0; i < TILE_NUM * BUFFER_NUM; i++) {
      LocalTensor<float> expLocal = expQueue.AllocTensor<float>();
      LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
      LocalTensor<float> broadcastTensor = inQueueX.AllocTensor<float>();

      // Читаем ранее сохраненные экспоненты
      DataCopy(expLocal, yGm[i * TILE_LENGTH], TILE_LENGTH);

      // Подготовка тензора-множителя (Broadcast)
      for (int32_t j = 0; j < TILE_LENGTH; j++) {
        broadcastTensor.SetValue(j, reciprocal);
      }

      // y = exp * (1/sum)
      Mul(yLocal, expLocal, broadcastTensor, TILE_LENGTH);

      // Записываем финальный результат
      DataCopy(yGm[i * TILE_LENGTH], yLocal, TILE_LENGTH);

      inQueueX.FreeTensor(broadcastTensor);
      outQueueY.FreeTensor(yLocal);
      expQueue.FreeTensor(expLocal);
    }
  }

 private:
  TPipe pipe;
  TQue<TPosition::VECIN, BUFFER_NUM> inQueueX;
  TQue<TPosition::VECOUT, BUFFER_NUM> expQueue;
  TQue<TPosition::VECOUT, BUFFER_NUM> outQueueY;
  TQue<TPosition::VECIN, 1> tmpQueue;

  GlobalTensor<float> xGm;
  GlobalTensor<float> yGm;
};

extern "C" __global__ __aicore__ void softmax_custom(GM_ADDR x, GM_ADDR y) {
  KernelSoftmax op;
  op.Init(x, y);
  op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
void softmax_custom_do(uint32_t blockDim, void* stream, uint8_t* x,
                       uint8_t* y) {
  softmax_custom<<<blockDim, nullptr, stream>>>(x, y);
}
#endif