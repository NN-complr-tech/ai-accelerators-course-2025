/**
 * @file softmax_custom.cpp
 * Модифицированный Softmax kernel из 4 лабы с константами из config.h
 *
 * Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "config.h"
#include "kernel_operator.h"

using namespace AscendC;

class KernelSoftmax {
 public:
  __aicore__ inline KernelSoftmax() {}

  __aicore__ inline void Init(GM_ADDR x, GM_ADDR z) {
    // Настройка смещения для каждого ядра
    xGm.SetGlobalBuffer((__gm__ float*)x + BLOCK_LENGTH * GetBlockIdx(),
                        BLOCK_LENGTH);
    zGm.SetGlobalBuffer((__gm__ float*)z + BLOCK_LENGTH * GetBlockIdx(),
                        BLOCK_LENGTH);

    // Инициализация очередей
    pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_LENGTH * sizeof(float));
    pipe.InitBuffer(expQueue, BUFFER_NUM, TILE_LENGTH * sizeof(float));
    pipe.InitBuffer(outQueueZ, BUFFER_NUM, TILE_LENGTH * sizeof(float));
    pipe.InitBuffer(tmpQueue, 1, TILE_LENGTH * sizeof(float));
  }

  __aicore__ inline void Process() {
    // 1. Находим максимум в блоке (для стабильности exp)
    float blockMax = FindBlockMax();

    // 2. Считаем сумму экспонент и сохраняем промежуточные exp(x - max)
    float totalSum = ComputeExpSumAndStore(blockMax);

    // 3. Финальная нормализация: z = exp / totalSum
    Normalize(totalSum);
  }

 private:
  // Поиск максимума во всем блоке
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
      LocalTensor<float> workLocal = outQueueZ.AllocTensor<float>();

      DataCopy(xLocal, xGm[i * TILE_LENGTH], TILE_LENGTH);

      // Стабилизация: x - max
      for (int32_t j = 0; j < TILE_LENGTH; j++) {
        workLocal.SetValue(j, blockMax);
      }
      Sub(expLocal, xLocal, workLocal, TILE_LENGTH);

      // exp(x - max)
      Exp(expLocal, expLocal, TILE_LENGTH);

      // Считаем сумму текущего тайла
      for (int32_t j = 0; j < TILE_LENGTH; j++) {
        totalSum += expLocal.GetValue(j);
      }

      // Сохраняем результат экспоненты временно в выходной буфер GM
      DataCopy(zGm[i * TILE_LENGTH], expLocal, TILE_LENGTH);

      outQueueZ.FreeTensor(workLocal);
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
      LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();
      LocalTensor<float> broadcastTensor = inQueueX.AllocTensor<float>();

      // Читаем ранее сохраненные экспоненты
      DataCopy(expLocal, zGm[i * TILE_LENGTH], TILE_LENGTH);

      // Подготовка тензора-множителя (Broadcast)
      for (int32_t j = 0; j < TILE_LENGTH; j++) {
        broadcastTensor.SetValue(j, reciprocal);
      }

      // z = exp * (1/sum)
      Mul(zLocal, expLocal, broadcastTensor, TILE_LENGTH);

      // Записываем финальный результат
      DataCopy(zGm[i * TILE_LENGTH], zLocal, TILE_LENGTH);

      inQueueX.FreeTensor(broadcastTensor);
      outQueueZ.FreeTensor(zLocal);
      expQueue.FreeTensor(expLocal);
    }
  }

 private:
  TPipe pipe;
  TQue<TPosition::VECIN, BUFFER_NUM> inQueueX;
  TQue<TPosition::VECOUT, BUFFER_NUM> expQueue;
  TQue<TPosition::VECOUT, BUFFER_NUM> outQueueZ;
  TQue<TPosition::VECIN, 1> tmpQueue;

  GlobalTensor<float> xGm;
  GlobalTensor<float> zGm;
};

extern "C" __global__ __aicore__ void softmax_custom(GM_ADDR x, GM_ADDR z) {
  KernelSoftmax op;
  op.Init(x, z);
  op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
void softmax_custom_do(uint32_t blockDim, void* stream, uint8_t* x,
                       uint8_t* z) {
  softmax_custom<<<blockDim, nullptr, stream>>>(x, z);
}
#endif