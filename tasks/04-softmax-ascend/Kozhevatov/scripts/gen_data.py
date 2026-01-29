#!/usr/bin/python3
# coding=utf-8
import numpy as np
import os

NUM_CORES = 8
BLOCK_SIZE = 2048 
TOTAL_SIZE = NUM_CORES * BLOCK_SIZE

def gen_golden_data_simple():
    # Создаем папки, если их нет
    os.makedirs("./input", exist_ok=True)
    os.makedirs("./output", exist_ok=True)

    # Генерируем вектор
    input_data = np.random.uniform(-5, 5, [TOTAL_SIZE]).astype(np.float32)
    
    # Вычисляем softmax по блокам (как делает ядро)
    golden = np.zeros_like(input_data)
    for core in range(NUM_CORES):
        offset = core * BLOCK_SIZE
        block = input_data[offset : offset + BLOCK_SIZE]
        
        max_val = np.max(block)
        exp_block = np.exp(block - max_val)
        golden[offset : offset + BLOCK_SIZE] = exp_block / np.sum(exp_block)
    
    # Сохраняем файлы
    input_data.tofile("./input/softmax_input.bin")
    golden.tofile("./output/golden.bin")
    print("Generated input/softmax_input.bin and output/golden.bin")

if __name__ == '__main__':
    gen_golden_data_simple()
