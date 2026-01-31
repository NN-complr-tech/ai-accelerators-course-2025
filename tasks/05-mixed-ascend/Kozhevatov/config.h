/**
 * @file config.h
 * Центральный файл конфигурации параметров матрицы и ядер.
 */
#ifndef CONFIG_H
#define CONFIG_H

#include <stdint.h>

// 1. Параметры геометрии матрицы
#define MATRIX_SIZE 96
#define TOTAL_LENGTH (MATRIX_SIZE * MATRIX_SIZE)  // 9216

// 2. Параметры аппаратного ускорения (Ядра)
// Для Softmax: используем по 1 ядру на каждую строку (всего 96 ядер)
#define USE_CORE_NUM 96
#define BLOCK_LENGTH (TOTAL_LENGTH / USE_CORE_NUM)  // 9216 / 96 = 96

// Для MatMul: 2 ядра делят матрицу 96x96 на куски по 48 столбцов
#define MATMUL_CORE_NUM 2

// 3. Параметры активации
#define LEAKY_RELU_ALPHA 0.001f

// 5. Константы для Softmax
#define BUFFER_NUM 2  // Двойная буферизация
#define TILE_NUM 1
#define TILE_LENGTH (BLOCK_LENGTH / TILE_NUM / BUFFER_NUM)  // 96 / 2 / 1 = 48

#endif  // CONFIG_H
