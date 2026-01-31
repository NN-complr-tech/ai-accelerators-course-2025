#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================

import numpy as np
import os

# ============================================================================
# КОНСТАНТЫ (соответствуют config.h)
# ============================================================================
MATRIX_SIZE = 96            # Размер стороны матрицы
TOTAL_LENGTH = MATRIX_SIZE * MATRIX_SIZE  # Общее количество элементов  
USE_CORE_NUM = MATRIX_SIZE  # Количество ядер (по одному на строку для softmax)

# Параметры LeakyReLU - ДОЛЖНО СОВПАДАТЬ С config.h!
LEAKY_RELU_ALPHA = 0.001

# Производные параметры
BLOCK_LENGTH = TOTAL_LENGTH // USE_CORE_NUM  # = MATRIX_SIZE (одна строка на ядро)

# ============================================================================

def softmax(matrix):
    """Вычисление softmax по строкам (axis=1) - каждая строка независимо"""
    max_vals = np.max(matrix, axis=1, keepdims=True)
    exp_data = np.exp(matrix - max_vals)
    return exp_data / np.sum(exp_data, axis=1, keepdims=True)

def gen_golden_data():
    """Генерация тестовых данных для matmul + leaky relu + softmax"""
    alpha = 0.001  # Коэффициент для масштабирования входных данных
    
    # Генерация входных матриц для matmul
    input_a = np.random.randint(-1, 1, [MATRIX_SIZE, MATRIX_SIZE]).astype(np.float16) * alpha
    input_b = np.random.randint(-1, 1, [MATRIX_SIZE, MATRIX_SIZE]).astype(np.float16) * alpha
    input_bias = np.random.randint(-1, 1, [MATRIX_SIZE]).astype(np.float32)
    
    # Golden для matmul + bias
    golden_matmul = (np.matmul(input_a.astype(np.float32), input_b.astype(np.float32)) + input_bias).astype(np.float32)
    
    # ПРИМЕНЯЕМ LeakyReLU (ВАЖНО: эта операция теперь выполняется!)
    golden_matmul = np.where(golden_matmul >= 0, golden_matmul, golden_matmul * LEAKY_RELU_ALPHA)
    
    # Golden для softmax (по строкам, axis=1)
    golden_softmax = softmax(golden_matmul)
    
    # Создание директорий
    os.makedirs("./input", exist_ok=True)
    os.makedirs("./output", exist_ok=True)
    
    # Сохранение входных данных
    input_a.tofile("./input/x1_gm.bin")
    input_b.tofile("./input/x2_gm.bin")
    input_bias.tofile("./input/bias.bin")
    
    # Сохранение эталонных результатов
    golden_matmul.tofile("./output/golden.bin")
    golden_softmax.tofile("./output/golden_softmax.bin")
    
    # Отладочная информация
    print(f"✓ Сгенерированы данные для матрицы {MATRIX_SIZE}x{MATRIX_SIZE}")
    print(f"✓ Используется {USE_CORE_NUM} ядер")
    print(f"✓ Элементов на ядро: {BLOCK_LENGTH}")
    print(f"✓ LeakyReLU alpha: {LEAKY_RELU_ALPHA}")
    print(f"✓ Первые 3 элемента matmul результата:")
    print(f"  {golden_matmul.flatten()[:3]}")
    print(f"✓ Первые 3 элемента softmax результата:")
    print(f"  {golden_softmax.flatten()[:3]}")

if __name__ == "__main__":
    gen_golden_data()