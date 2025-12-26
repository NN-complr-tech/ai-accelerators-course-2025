import numpy as np
import os
import argparse

def gen_golden_data():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, required=True, help="matrix size N x N")
    
    args = parser.parse_args()

    N = args.N

    input_a = np.random.randint(1, 10, [N, N]).astype(np.float16)
    input_b = np.random.randint(1, 10, [N, N]).astype(np.float16)

    # alpha = 0.001
    
    golden = (np.matmul(input_a.astype(np.float32), input_b.astype(np.float32))).astype(np.float32)
    
    # olden = np.where(golden >= 0, golden, golden * alpha)
    
    os.system("mkdir -p input")
    os.system("mkdir -p output")
    
    input_a.tofile("./input/A.bin")
    input_b.tofile("./input/B.bin")

    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data()
