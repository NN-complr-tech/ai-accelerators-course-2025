import numpy as np
import argparse

def calc_softmax(x):
    e_x = np.exp(x)
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def gen_golden_data_simple():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, required=True, help="matrix size N x N")
    
    args = parser.parse_args()

    N = args.N

    input_x = np.random.uniform(-1, 1, (N, N)).astype(np.float32)
    golden = calc_softmax(input_x)

    input_x.reshape(-1).tofile("./input/input_x.bin")
    golden.reshape(-1).tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()
