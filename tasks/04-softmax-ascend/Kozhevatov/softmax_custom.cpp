/**
 * @file softmax_custom.cpp
 * @brief Реализация векторного ядра Softmax для Huawei Ascend NPU с
 * использованием Ascend C
 *
 * Данный файл содержит оптимизированную реализацию оператора Softmax для матриц
 * 64×64 с использованием фреймворка Ascend C. Реализация использует:
 * 1. Параллелизм на уровне блоков (16 блоков)
 * 2. Двойную буферизацию для перекрытия вычислений и передачи данных
 * 3. Векторизацию через Ascend C API (Exp, ReduceSum, Broadcast, Div)
 * 4. Двухпроходный алгоритм: вычисление суммы экспонент и нормализация
 *
 * @note Ascend C built-in softmax function НЕ используется (требование задания)
 *
 * Основные этапы вычислений:
 * 1. Инициализация глобальной памяти и локальных буферов
 * 2. Построчное вычисление Softmax:
 *    - Первый проход: вычисление суммы экспонент по строке
 *    - Второй проход: нормализация (exp(x)/sum)
 * 3. Запись результатов в глобальную память
 *
 * Архитектурные особенности:
 * - Каждый блок обрабатывает 4 строки матрицы (64 строк / 16 блоков)
 * - Каждая строка делится на тайлы для эффективного использования локальной
 * памяти
 * - Используется паттерн "double buffering" для скрытия латентности памяти
 */

#include "kernel_operator.h"

// Константы, определяющие размер задачи и схему параллелизма
constexpr long BLOCK_DIM =
    16;  ///< Количество блоков (ядер) для параллельной обработки
constexpr long SIZE = 64;  ///< Размер квадратной матрицы (SIZE x SIZE)

/// Общее количество элементов в матрице
constexpr int32_t TOTAL_LENGTH = SIZE * SIZE;

/// Количество строк, обрабатываемых каждым блоком
constexpr int32_t ROWS_PER_BLOCK = SIZE / BLOCK_DIM;

/// Длина данных, обрабатываемая каждым ядром (одна строка = 64 элемента)
constexpr int32_t BLOCK_LENGTH = SIZE;

/// Количество тайлов на блок для эффективного использования памяти
constexpr int32_t TILE_NUM = 2;

/// Количество буферов в каждой очереди (двойная буферизация)
constexpr int32_t BUFFER_NUM = 2;

/// Размер одного тайла с учетом двойной буферизации
constexpr int32_t TILE_LENGTH = BLOCK_LENGTH / TILE_NUM / BUFFER_NUM;

/**
 * @class KernelSoftmax
 * @brief Основной класс, реализующий ядро Softmax на Ascend C
 *
 * Класс инкапсулирует всю логику вычисления Softmax с использованием:
 * - Глобальной памяти для входных/выходных данных
 * - Локальной памяти (очереди и буферы) для промежуточных вычислений
 * - Векторных операций Ascend C API
 */
class KernelSoftmax {
 public:
  /**
   * @brief Конструктор по умолчанию
   * @note Выполняется на устройстве (декоратор __aicore__)
   */
  __aicore__ inline KernelSoftmax() {}

  /**
   * @brief Инициализация ресурсов ядра
   * @param x Указатель на входные данные в глобальной памяти
   * @param z Указатель на выходные данные в глобальной памяти
   *
   * Выполняет:
   * 1. Настройку глобальных тензоров с учетом индекса блока
   * 2. Инициализацию очередей ввода/вывода с двойной буферизацией
   * 3. Выделение временных буферов для промежуточных вычислений
   */
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR z) {
    // Инициализация глобальных тензоров с учетом смещения для каждого блока
    xGm.SetGlobalBuffer((__gm__ float*)x + BLOCK_LENGTH * ROWS_PER_BLOCK *
                                               AscendC::GetBlockIdx(),
                        BLOCK_LENGTH * BLOCK_DIM);
    zGm.SetGlobalBuffer((__gm__ float*)z + BLOCK_LENGTH * ROWS_PER_BLOCK *
                                               AscendC::GetBlockIdx(),
                        BLOCK_LENGTH * BLOCK_DIM);

    // Инициализация очередей с двойной буферизацией
    pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_LENGTH * sizeof(float));
    pipe.InitBuffer(outQueueZ, BUFFER_NUM, TILE_LENGTH * sizeof(float));

    // Инициализация временных буферов для вычислений
    pipe.InitBuffer(tmpBufCalc, TILE_LENGTH * sizeof(float));
    pipe.InitBuffer(tmpBuf, TILE_LENGTH * sizeof(float));
  }

  /**
   * @brief Основной метод обработки данных
   *
   * Реализует двухпроходный алгоритм Softmax:
   * 1. Первый проход: вычисление суммы экспонент по строке
   * 2. Второй проход: нормализация каждого элемента
   *
   * Для каждой строки выполняет:
   * - Копирование тайлов во входную очередь
   * - Вычисление экспонент и их суммирование
   * - Редукцию суммы по строке и broadcast для нормализации
   * - Вычисление нормализованных значений и запись в выходную очередь
   */
  __aicore__ inline void Process() {
    // Получение локальных тензоров из временных буферов
    tmpCalc = tmpBufCalc.Get<float>();
    tmp = tmpBuf.Get<float>();

    // Обработка строк, назначенных текущему блоку
    for (int32_t row = 0; row < ROWS_PER_BLOCK; row++) {
      int32_t row_offset = row * BLOCK_LENGTH;

      // Инициализация аккумулятора суммы нулями
      AscendC::Duplicate(tmpCalc, 0.0f, TILE_LENGTH);

      // ПЕРВЫЙ ПРОХОД: вычисление суммы экспонент по строке
      for (uint32_t i = 0; i < TILE_NUM * BUFFER_NUM; i++) {
        CopyIn(i, row_offset);  // Копирование тайла
        ComputeSum(i);          // Добавление экспоненты к сумме
      }

      // Редукция: получение итоговой суммы по строке
      constexpr uint32_t shape[] = {1, TILE_LENGTH};
      AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR>(tmp, tmpCalc,
                                                              shape, true);

      // Broadcast: распространение суммы на все элементы для нормализации
      constexpr uint32_t dst_shape[] = {TILE_LENGTH};
      constexpr uint32_t src_shape[] = {1};
      AscendC::Broadcast<float, 1, 0>(tmpCalc, tmp, dst_shape, src_shape);

      // ВТОРОЙ ПРОХОД: вычисление нормализованных значений
      for (uint32_t i = 0; i < TILE_NUM * BUFFER_NUM; i++) {
        CopyIn(i, row_offset);   // Повторное копирование (данные те же)
        ComputeSoftmax(i);       // Вычисление exp(x)/sum
        CopyOut(i, row_offset);  // Запись результата
      }
    }
  }

 private:
  /**
   * @brief Копирование тайла данных из глобальной в локальную память
   * @param progress Индекс тайла в текущей строке
   * @param row_offset Смещение текущей строки в глобальной памяти
   *
   * Использует паттерн producer-consumer с двойной буферизацией:
   * 1. Выделение локального тензора из очереди
   * 2. Копирование данных из глобальной памяти
   * 3. Помещение тензора в очередь для обработки
   */
  __aicore__ inline void CopyIn(int32_t progress, int32_t row_offset) {
    AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
    AscendC::DataCopy(xLocal, xGm[row_offset + progress * TILE_LENGTH],
                      TILE_LENGTH);
    inQueueX.EnQue(xLocal);
  }

  /**
   * @brief Вычисление экспоненты и добавление к аккумулятору суммы
   * @param progress Индекс тайла в текущей строке
   *
   * Выполняет:
   * 1. Извлечение тайла из очереди
   * 2. Вычисление экспоненты каждого элемента
   * 3. Добавление результата к аккумулятору tmpCalc
   * 4. Освобождение локального тензора
   */
  __aicore__ inline void ComputeSum(int32_t progress) {
    AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
    AscendC::Exp(xLocal, xLocal, TILE_LENGTH);  // Вычисление exp(x)

    AscendC::Add(tmpCalc, tmpCalc, xLocal,
                 TILE_LENGTH);  // Аккумулирование суммы

    inQueueX.FreeTensor(xLocal);  // Возврат памяти в очередь
  }

  /**
   * @brief Вычисление нормализованного значения (exp(x)/sum)
   * @param progress Индекс тайла в текущей строке
   *
   * Выполняет:
   * 1. Извлечение входных данных из очереди
   * 2. Выделение выходного тензора
   * 3. Вычисление exp(x) и деление на сумму (tmpCalc)
   * 4. Помещение результата в выходную очередь
   */
  __aicore__ inline void ComputeSoftmax(int32_t progress) {
    AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
    AscendC::LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();

    AscendC::Exp(xLocal, xLocal, TILE_LENGTH);           // exp(x)
    AscendC::Div(zLocal, xLocal, tmpCalc, TILE_LENGTH);  // exp(x)/sum

    outQueueZ.EnQue<float>(zLocal);  // Результат для копирования
    inQueueX.FreeTensor(xLocal);     // Освобождение входных данных
  }

  /**
   * @brief Копирование результата из локальной в глобальную память
   * @param progress Индекс тайла в текущей строке
   * @param row_offset Смещение текущей строки в глобальной памяти
   */
  __aicore__ inline void CopyOut(int32_t progress, int32_t row_offset) {
    AscendC::LocalTensor<float> zLocal = outQueueZ.DeQue<float>();
    AscendC::DataCopy(zGm[row_offset + progress * TILE_LENGTH], zLocal,
                      TILE_LENGTH);
    outQueueZ.FreeTensor(zLocal);  // Освобождение выходного тензора
  }

 private:
  // Структуры данных Ascend C для управления памятью и вычислениями
  AscendC::TPipe pipe;  ///< Трубопровод для управления памятью
  AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM>
      inQueueX;  ///< Входная очередь с двойной буферизацией
  AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM>
      outQueueZ;  ///< Выходная очередь с двойной буферизацией
  AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf,
      tmpBufCalc;                    ///< Временные буферы для вычислений
  AscendC::GlobalTensor<float> xGm;  ///< Глобальный тензор входных данных
  AscendC::GlobalTensor<float> zGm;  ///< Глобальный тензор выходных данных
  AscendC::LocalTensor<float>
      tmpCalc;  ///< Локальный тензор для аккумулятора суммы
  AscendC::LocalTensor<float>
      tmp;  ///< Локальный тензор для промежуточных результатов
};

/**
 * @brief Точка входа ядра Ascend C
 * @param x Указатель на входные данные в глобальной памяти
 * @param z Указатель на выходные данные в глобальной памяти
 *
 * Данная функция выполняется на устройстве и запускает обработку данных:
 * 1. Создание экземпляра ядра
 * 2. Инициализация ресурсов
 * 3. Запуск процесса вычислений
 *
 * @note Декораторы указывают на выполнение на AI Core:
 *       __global__ - функция ядра
 *       __aicore__ - выполнение на AI Core
 */
extern "C" __global__ __aicore__ void softmax_custom(GM_ADDR x, GM_ADDR z) {
  KernelSoftmax op;
  op.Init(x, z);
  op.Process();
}

/**
 * @brief Host-функция для запуска ядра на устройстве
 * @param blockDim Количество блоков для запуска
 * @param stream Поток выполнения (CUDA/Hip stream)
 * @param x Указатель на входные данные на устройстве
 * @param z Указатель на выходные данные на устройстве
 *
 * Данная функция предназначена для запуска в режиме NPU (не CPU debug)
 * и выполняет:
 * 1. Запуск ядра с заданным количеством блоков
 * 2. Привязку к потоку выполнения
 */
#ifndef ASCENDC_CPU_DEBUG
void softmax_custom_do(uint32_t blockDim, void* stream, uint8_t* x,
                       uint8_t* z) {
  softmax_custom<<<blockDim, nullptr, stream>>>(x, z);
}
#endif