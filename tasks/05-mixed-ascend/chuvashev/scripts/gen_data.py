import numpy as np
import os


def gen_golden_data():
    M = 1024
    N = 640
    K = 256

    input_a = np.random.randint(1, 10, [M, K]).astype(np.float16)
    input_b = np.random.randint(1, 10, [K, N]).astype(np.float16)

    alpha = 0.001
    
    golden = (np.matmul(input_a.astype(np.float32), input_b.astype(np.float32))).astype(np.float32)
    
    golden = np.where(golden >= 0, golden, golden * alpha)
    
    os.system("mkdir -p input")
    os.system("mkdir -p output")
    
    input_a.tofile("./input/A.bin")
    input_b.tofile("./input/B.bin")

    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data()
