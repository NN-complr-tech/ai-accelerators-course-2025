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

struct timer {
  timer() = delete;
  timer(const char *msg, bool q = true);

  ~timer();

  void reset();
  double elapsed();

  std::chrono::high_resolution_clock::time_point tStart;
  std::string message;
  bool quiet;
};

timer::timer(const char *msg, bool q) : message(msg), quiet(q) {
  tStart = std::chrono::high_resolution_clock::now();
}

double timer::elapsed() {
  return std::chrono::duration_cast<std::chrono::duration<double>>(
             std::chrono::high_resolution_clock::now() - tStart)
      .count();
}

void timer::reset() { tStart = std::chrono::high_resolution_clock::now(); }

timer::~timer() {
  if (!message.empty()) {
    if (!quiet) std::cout << message << ' ' << elapsed() << " s\n";
  }
}

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

namespace {
void make_matrix(std::size_t n, std::vector<float> &matrix) {
  // throw std::runtime_error("make_matrix not implemented");
  std::random_device rd;
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::mt19937 gen(rd());
  std::size_t size = n * n;
#pragma omp parallel for num_threads(12)
  std::cout << omp_get_thread_num() << std::endl;
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

__global__ void warmup_kernel(const float *d_input, float *d_output,
                              const std::size_t n) {
  std::size_t row = blockIdx.y;
  std::size_t col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < n && col < n) {
    std::size_t index_of_start_elem = row * n + col;

    d_output[index_of_start_elem] = d_input[index_of_start_elem];
  }
}

void warmup_cuda(const std::vector<float> &matrix, std::size_t n) {
  // throw std::runtime_error("CUDA warm-up not implemented");

  cudaSetDevice(0);

  std::size_t byte_size_memory = n * n * sizeof(float);
  float *d_input, *d_output, *d_sum_of_row;

  cudaMalloc(&d_input, byte_size_memory);
  cudaMalloc(&d_output, byte_size_memory);
  cudaMalloc(&d_sum_of_row, n * sizeof(float));
  cudaMemcpy(d_input, matrix.data(), byte_size_memory, cudaMemcpyHostToDevice);

  dim3 threds_per_block(512);
  dim3 blocks((n + (threds_per_block.x - 1)) / threds_per_block.x, n);
  warmup_kernel<<<blocks, threds_per_block>>>(d_input, d_output, n);

  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_sum_of_row);
}

__global__ void softmax_exp_sum_kernel(float *d_matrix,
                                       float *d_sum_of_row, size_t n) {
  size_t row = blockIdx.y;
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < n && col < n) {
    size_t index = row * n + col;
    float val = expf(d_matrix[index]);
    d_matrix[index] = val;
    atomicAdd(&d_sum_of_row[row], val);
  }
}

__global__ void softmax_divide_kernel(float *d_matrix,
                                      const float *d_sum_of_row, size_t n) {
  size_t row = blockIdx.y;
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < n && col < n) {
    size_t index = row * n + col;
    d_matrix[index] /= d_sum_of_row[row];
  }
}

void launch_softmax_kernel(float *d_matrix, float *d_sum_of_row,
                           size_t n) {
  dim3 threads_per_block(512);
  dim3 blocks((n + threads_per_block.x - 1) / threads_per_block.x, n);

  timer first("");
  softmax_exp_sum_kernel<<<blocks, threads_per_block>>>(d_matrix, d_sum_of_row,
                                                        n);
  cudaDeviceSynchronize();
  double first_end = first.elapsed();
  std::cout << "first time: " << first_end << "\t GB/s = "
            << n * n * 4.0 * 2.0 / (1024.0 * 1024.0 * 1024.0 * first_end)
            << std::endl;

  timer second("");
  softmax_divide_kernel<<<blocks, threads_per_block>>>(d_matrix, d_sum_of_row,
                                                       n);
  cudaDeviceSynchronize();
  double secod_end = second.elapsed();
  std::cout << "second time: " << secod_end << "\t GB/s = "
            << n * n * 4.0 * 2.0 / (1024.0 * 1024.0 * 1024.0 * secod_end)
            << std::endl;
}

void run_cuda_simt(const std::vector<float> &input, std::vector<float> &output,
                   std::size_t n) {
  cudaSetDevice(0);

  size_t byte_size = n * n * sizeof(float);

  float *d_matrix;
  float *d_sum_of_row;

  cudaMalloc(&d_matrix, byte_size);
  cudaMalloc(&d_sum_of_row, n * sizeof(float));

  cudaMemset(d_sum_of_row, 0, n * sizeof(float));

  cudaMemcpy(d_matrix, input.data(), byte_size, cudaMemcpyHostToDevice);

  timer time("");
  launch_softmax_kernel(d_matrix, d_sum_of_row, n);
  std::cout << "timer: " << time.elapsed() << std::endl;

  cudaMemcpy(output.data(), d_matrix, byte_size, cudaMemcpyDeviceToHost);

  cudaFree(d_matrix);
  cudaFree(d_sum_of_row);
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
    /*if (argc != 2) {
      std::cerr << "Usage: " << argv[0] << " <matrix_size_n>\n";
      return EXIT_FAILURE;
    }

    const std::size_t n = static_cast<std::size_t>(std::stoul(argv[1]));
    if (n == 0) {
      throw std::invalid_argument("Matrix size must be positive");
    }*/

    std::size_t n = 20000;

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
