#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

void make_input_matrix(std::vector<__half> &matrix, std::size_t n) {
  // throw std::runtime_error("make_input_matrix not implemented");
  std::random_device rd;
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::mt19937 gen(rd());
#pragma omp parallel for
  for (int idx = 0; idx < n * n; ++idx) {
    matrix[idx] = __float2half(dist(gen));
  }
}

void run_openmp_reference(const std::vector<__half> &input,
                          std::vector<float> &output, const std::size_t n) {
  // throw std::runtime_error("OpenMP reference not implemented");

  int block_size = 32;
  int count_of_blocks = (n + block_size - 1) / block_size;

#pragma omp parallel for
  for (int ii = 0; ii < count_of_blocks; ++ii) {
    int i_start = ii * block_size;
    int i_end = (std::min)((ii + 1) * block_size, (int)n);
    for (int jj = 0; jj < count_of_blocks; ++jj) {
      int j_start = jj * block_size;
      int j_end = (std::min)((jj + 1) * block_size, (int)n);
      for (int kk = 0; kk < count_of_blocks; ++kk) {
        int k_start = kk * block_size;
        int k_end = (std::min)((kk + 1) * block_size, (int)n);
        for (int i = i_start; i < i_end; ++i) {
          for (int k = k_start; k < k_end; ++k) {
            float value = __half2float(input[i * n + k]);
            for (int j = j_start; j < j_end; ++j) {
              output[i * n + j] += value * __half2float(input[k * n + j]);
            }
          }
        }
      }
    }
  }
}
