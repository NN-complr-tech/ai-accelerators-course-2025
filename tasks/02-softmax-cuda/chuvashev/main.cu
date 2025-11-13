#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <ratio>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

// TODO: Move to utils
#define CHECK_CUDA_ERROR(callable)                                        \
  {                                                                       \
    auto codeError = callable;                                            \
    if (codeError != cudaSuccess) {                                       \
      std::cerr << "\033[1;31merror\033[0m: ";                            \
      std::cerr << cudaGetErrorString(codeError) << '\n';                 \
      std::cerr << "code error: " << static_cast<int>(codeError) << '\n'; \
      std::cerr << "loc: " << __FILE__ << '(' << __LINE__ << ")\n";       \
      std::exit(codeError);                                               \
    }                                                                     \
  }

#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 1024
#endif

struct timer {
  std::chrono::high_resolution_clock::time_point t_start;

  timer() { t_start = std::chrono::high_resolution_clock::now(); }
  ~timer() {}

  void reset() { t_start = std::chrono::high_resolution_clock::now(); }
  double elapsed() {
    return std::chrono::duration_cast<std::chrono::duration<double>>(
               std::chrono::high_resolution_clock::now() - t_start)
        .count();
  }
};

namespace {
void make_matrix(std::size_t n, std::vector<float> &matrix) {
  // throw std::runtime_error("make_matrix not implemented");
  std::random_device rd;
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::mt19937 gen(rd());
  std::size_t size = n * n;
#pragma omp parallel for num_threads(12)
  for (std::size_t idx = 0; idx < size; ++idx) {
    matrix[idx] = dist(gen);
  }
}

void print_matrix(const std::size_t n, const std::vector<float> &matrix) {
  for (std::size_t idx_i = 0; idx_i < n; ++idx_i) {
    for (std::size_t idx_j = 0; idx_j < n; ++idx_j) {
      std::cout << matrix[idx_i * n + idx_j] << "\t";
    }
    std::cout << "\n";
  }
}

void run_sequential(const std::vector<float> &input, std::vector<float> &output,
                    std::size_t n) {
  // throw std::runtime_error("Sequential method not implemented");

  for (std::size_t idx_i = 0; idx_i < n; ++idx_i) {
    float current_sum = 0.0f;
    for (std::size_t idx_j = 0; idx_j < n; ++idx_j) {
      float current_value = std::exp(input[idx_i * n + idx_j]);
      current_sum += current_value;
      output[idx_i * n + idx_j] = current_value;
    }
    current_sum = 1.0f / current_sum;
    for (std::size_t idx_j = 0; idx_j < n; ++idx_j) {
      output[idx_i * n + idx_j] *= current_sum;
    }
  }
}

__global__ void warmup_kernel(float *d_matrix, const std::size_t n) {
  std::size_t row = blockIdx.y;
  std::size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < n && col < n) {
    std::size_t index_of_start_elem = row * n + col;
    d_matrix[index_of_start_elem] = 2.0f * d_matrix[index_of_start_elem];
  }
}

void warmup_cuda(const std::vector<float> &matrix, std::size_t n) {
  // throw std::runtime_error("CUDA warm-up not implemented");
  CHECK_CUDA_ERROR(cudaSetDevice(0));
  std::size_t byte_size_memory = n * n * sizeof(float);
  float *d_matrix;
  CHECK_CUDA_ERROR(cudaMalloc(&d_matrix, byte_size_memory));
  CHECK_CUDA_ERROR(cudaMemcpy(d_matrix, matrix.data(), byte_size_memory,
                              cudaMemcpyHostToDevice));
  dim3 threds_per_block(1024);
  dim3 blocks(n / 1024, n);
  //timer timer;
  warmup_kernel<<<blocks, threds_per_block>>>(d_matrix, n);
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());
  /*double time = timer.elapsed();
  std::cout << time << "\tGB/s: "
            << n * n * sizeof(float) * 2.0f /
                   (1024.0f * 1024.0f * 1024.0f * time)
            << std::endl;*/
  CHECK_CUDA_ERROR(cudaFree(d_matrix));
}

__global__ void softmax_kernel(float *d_matrix, size_t n) {
  size_t row = blockIdx.x;

  __shared__ float smem[THREADS_PER_BLOCK];
  smem[threadIdx.x] = 0.0f;
  __syncthreads();

  if (row < n) {
    for (std::size_t col = threadIdx.x; col < n; col += blockDim.x) {
      size_t index = row * n + col;
      smem[threadIdx.x] += expf(d_matrix[index]);
    }
  }

  __syncthreads();

  /*if (threadIdx.x == 0) {
    float local_sum = 0.0f;
    for (std::size_t idx = 0; idx < THREADS_PER_BLOCK; ++idx) {
      local_sum += smem[idx];
    }
    smem[0] = local_sum;
  }
  __syncthreads();*/

  for (std::size_t s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      smem[threadIdx.x] += smem[threadIdx.x + s];
    }
    __syncthreads();
  }

  if (row < n) {
    for (std::size_t col = threadIdx.x; col < n; col += blockDim.x) {
      size_t index = row * n + col;
      d_matrix[index] = expf(d_matrix[index]) / smem[0];
    }
  }
}

void launch_softmax_kernel(float *d_matrix, size_t n) {
  dim3 threads_per_block(THREADS_PER_BLOCK);
  dim3 blocks(n);

  timer timer;
  softmax_kernel<<<blocks, threads_per_block>>>(d_matrix, n);
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());
  double time = timer.elapsed();
  /*std::cout << "CUDA: " << time << "\tGB/s: "
            << n * n * sizeof(float) * 3.0f /
                   (1024.0 * 1024.0f * 1024.0f * time)
            << std::endl;*/

  std::cout << time << std::endl;
  std::cout << "GB/s: "
            << n * n * sizeof(float) * 3.0f /
                   (1024.0 * 1024.0f * 1024.0f * time)
            << std::endl;
}

void run_cuda_simt(const std::vector<float> &input, std::vector<float> &output,
                   std::size_t n) {
  CHECK_CUDA_ERROR(cudaSetDevice(0));

  size_t byte_size = n * n * sizeof(float);
  float *d_matrix;

  CHECK_CUDA_ERROR(cudaMalloc(&d_matrix, byte_size));
  CHECK_CUDA_ERROR(
      cudaMemcpy(d_matrix, input.data(), byte_size, cudaMemcpyHostToDevice));

  launch_softmax_kernel(d_matrix, n);

  CHECK_CUDA_ERROR(
      cudaMemcpy(output.data(), d_matrix, byte_size, cudaMemcpyDeviceToHost));
  CHECK_CUDA_ERROR(cudaFree(d_matrix));
}

double measure_seconds(const std::function<void()> &work) {
  const auto start = std::chrono::high_resolution_clock::now();
  work();
  const auto stop = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double>(stop - start).count();
}

float max_abs_diff(const std::vector<float> &baseline,
                   const std::vector<float> &candidate) {
  if (baseline.size() != candidate.size()) {
    throw std::runtime_error(
        "Result size mismatch while validating correctness");
  }
  float max_diff = 0.0f;
  for (std::size_t i = 0; i < baseline.size(); ++i) {
    max_diff = std::max(max_diff, std::abs(baseline[i] - candidate[i]));
  }
  return max_diff;
}

// TODO: Create basic utils file
struct RunResult {
  std::vector<float> result;
  double seconds = 0.0;
  float diff = 0.0f;
  bool success = false;
  explicit operator bool() const noexcept { return success; }
};

std::string format_time(double seconds) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2) << seconds;
  return oss.str();
}

std::string format_diff(float diff) {
  std::ostringstream oss;
  oss << std::defaultfloat << std::setprecision(1) << diff;
  return oss.str();
}

void print_report(std::string_view testName, const RunResult &result) {
  if (result) {
    std::cout << testName << ": " << format_time(result.seconds)
              << " sec (diff: " << format_diff(result.diff) << ")\n";
  } else {
    std::cout << testName << ": n/a (diff: n/a)\n";
  }
}
}  // namespace

int main(int argc, char *argv[]) {
  try {
    if (argc != 2) {
      std::cerr << "Usage: " << argv[0] << " <matrix_size_n>\n";
      return EXIT_FAILURE;
    }

    const std::size_t n = static_cast<std::size_t>(std::stoul(argv[1]));
    if (n == 0) {
      throw std::invalid_argument("Matrix size must be positive");
    }

    std::vector<float> input(n * n, 0);
    make_matrix(n, input);

    std::vector<float> sequential_result(n * n, 0);
    const double sequential_seconds = measure_seconds(
        [&]() { return run_sequential(input, sequential_result, n); });
    // print_matrix(n, sequential_result);

    RunResult simt_res;
    simt_res.result.resize(n * n, 0);
    try {
      warmup_cuda(input, n);
      simt_res.seconds = measure_seconds(
          [&]() { return run_cuda_simt(input, simt_res.result, n); });
      simt_res.diff = max_abs_diff(sequential_result, simt_res.result);
      simt_res.success = true;
      // print_matrix(n, simt_res.result);
      // TODO: Compare simt_seconds with the OpenMP+AVX2 timing from practice
      // #1.
    } catch (const std::exception &ex) {
      std::cerr << "CUDA SIMT method failed: " << ex.what() << '\n';
    }

    std::cout << "Sequential: " << format_time(sequential_seconds) << " sec\n";
    print_report("SIMT", simt_res);

    return EXIT_SUCCESS;
  } catch (const std::exception &ex) {
    std::cerr << "Error: " << ex.what() << '\n';
  } catch (...) {
    std::cerr << "Unknown error\n";
  }

  return EXIT_FAILURE;
}
