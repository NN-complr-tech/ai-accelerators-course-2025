#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <omp.h>

#include <algorithm>
#include <array>
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

#include <cutlass/gemm/device/gemm.h>

#define WARP_SIZE 32
#define WMMA_SIZE 16
using namespace nvcuda;

#define BLOCK_SIZE 32

struct timer {
  std::chrono::high_resolution_clock::time_point t_start;
  timer();
  ~timer();
  void reset();
  double elapsed();
};

std::chrono::high_resolution_clock::time_point t_start;
timer::timer() { t_start = std::chrono::high_resolution_clock::now(); }
timer::~timer() {}
void timer::reset() { t_start = std::chrono::high_resolution_clock::now(); }
double timer::elapsed() {
  return std::chrono::duration_cast<std::chrono::duration<double>>(
             std::chrono::high_resolution_clock::now() - t_start)
      .count();
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

void make_input_matrix(std::vector<__half> &matrix, std::size_t n);
template <typename T>
void print_matrix(const std::vector<T> &matrix, const std::size_t n) {
  for (std::size_t idx_i = 0; idx_i < n; ++idx_i) {
    for (std::size_t idx_j = 0; idx_j < n; ++idx_j) {
      std::cout << (float)(matrix[idx_i * n + idx_j]) << "\t";
    }
    std::cout << "\n";
  }
}
void run_openmp_reference(const std::vector<__half> &input,
                          std::vector<float> &output, const std::size_t n);

__global__ void GPU_MATMUL_V1(const __half *input, float *output,
                              const std::size_t n) {
  std::size_t i = blockIdx.y * blockDim.y + threadIdx.y;
  std::size_t j = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= n || j >= n) return;

  float sum = 0.0f;

  for (std::size_t idx = 0; idx < n; ++idx) {
    sum += __half2float(input[i * n + idx]) * __half2float(input[idx * n + j]);
  }

  output[i * n + j] = sum;
}

void run_matrix_mult_gpu_ver_1(const std::vector<__half> &input,
                               std::vector<float> &output,
                               const std::size_t n) {
  cudaSetDevice(0);

  __half *d_input;
  float *d_output;

  cudaMalloc(&d_input, sizeof(__half) * n * n);
  cudaMalloc(&d_output, sizeof(float) * n * n);

  cudaMemcpy(d_input, input.data(), sizeof(__half) * n * n,
             cudaMemcpyHostToDevice);

  dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid_size((n + block_size.x - 1) / block_size.x,
                 (n + block_size.y - 1) / block_size.y);

  GPU_MATMUL_V1<<<grid_size, block_size>>>(d_input, d_output, n);

  cudaDeviceSynchronize();

  cudaMemcpy(output.data(), d_output, sizeof(float) * n * n,
             cudaMemcpyDeviceToHost);

  cudaFree(d_input);
  cudaFree(d_output);
}

__global__ void GPU_MATMUL_V2(const __half *input, float *output,
                              const std::size_t n) {
  std::size_t block_row = blockIdx.y;
  std::size_t block_col = blockIdx.x;

  std::size_t local_row = threadIdx.y;
  std::size_t local_col = threadIdx.x;

  std::size_t row = block_row * blockDim.y + local_row;
  std::size_t col = block_col * blockDim.x + local_col;

  if (row >= n || col >= n) return;

  __shared__ float block_a[BLOCK_SIZE * BLOCK_SIZE];
  __shared__ float block_b[BLOCK_SIZE * BLOCK_SIZE];

  float sum = 0.0f;

  for (std::size_t block = 0; block < gridDim.x; ++block) {
    block_a[local_row * BLOCK_SIZE + local_col] =
        (input[row * n + block * BLOCK_SIZE + local_col]);
    block_b[local_row * BLOCK_SIZE + local_col] =
        (input[col + (block * BLOCK_SIZE + local_row) * n]);

    __syncthreads();

    for (std::size_t k = 0; k < BLOCK_SIZE; ++k) {
      sum += (block_a[local_row * BLOCK_SIZE + k]) *
             (block_b[k * BLOCK_SIZE + local_col]);
    }

    __syncthreads();
  }

  output[row * n + col] = sum;
}

void run_matrix_mult_gpu_ver_2(const std::vector<__half> &input,
                               std::vector<float> &output,
                               const std::size_t n) {
  cudaSetDevice(0);

  __half *d_input;
  float *d_output;

  cudaMalloc(&d_input, sizeof(__half) * n * n);
  cudaMalloc(&d_output, sizeof(float) * n * n);

  cudaMemcpy(d_input, input.data(), sizeof(__half) * n * n,
             cudaMemcpyHostToDevice);

  dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid_size((n + block_size.x - 1) / block_size.x,
                 (n + block_size.y - 1) / block_size.y);

  timer timer;
  GPU_MATMUL_V2<<<grid_size, block_size>>>(d_input, d_output, n);
  cudaDeviceSynchronize();
  double end_time = timer.elapsed();
  std::cout << end_time << "\t"
            << "GB\s: "
            << float(n * n * sizeof(float) +
                     n * n * float((n + block_size.x - 1) / block_size.x) * 2 *
                         sizeof(__half)) /
                   (1024.0f * 1024.0f * 1024.0f * end_time)
            << std::endl;

  cudaMemcpy(output.data(), d_output, sizeof(float) * n * n,
             cudaMemcpyDeviceToHost);

  cudaFree(d_input);
  cudaFree(d_output);
}

void warmup_wmma(const std::vector<__half> &matrix, std::size_t n) {
  throw std::runtime_error("WMMA warm-up not implemented");
}

__global__ void WMMA_kernel(const __half *input, float *output,
                            const std::size_t n) {
  int warp_i = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
  int warp_j = (blockIdx.y * blockDim.y + threadIdx.y);

  wmma::fragment<wmma::matrix_a, WMMA_SIZE, WMMA_SIZE, WMMA_SIZE, __half,
                 wmma::row_major>
      a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_SIZE, WMMA_SIZE, WMMA_SIZE, __half,
                 wmma::row_major>
      b_frag;
  wmma::fragment<wmma::accumulator, WMMA_SIZE, WMMA_SIZE, WMMA_SIZE, float>
      acc_frag;

  wmma::fill_fragment(acc_frag, 0.0f);

  for (std::size_t k = 0; k < n; k += WMMA_SIZE) {
    int a_row = warp_i * WMMA_SIZE;
    int a_col = k;
    int b_row = k;
    int b_col = warp_j * WMMA_SIZE;

    wmma::load_matrix_sync(a_frag, input + a_row * n + a_col, n);
    wmma::load_matrix_sync(b_frag, input + b_row * n + b_col, n);

    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
  }

  int c_row = warp_i * WMMA_SIZE;
  int c_col = warp_j * WMMA_SIZE;

  wmma::store_matrix_sync(output + c_row * n + c_col, acc_frag, n,
                          wmma::mem_row_major);
}

void run_wmma(const std::vector<__half> &input, std::vector<float> &output,
              const std::size_t n) {
  // throw std::runtime_error("WMMA method not implemented");

  __half *d_input;
  float *d_output;

  cudaMalloc(&d_input, n * n * sizeof(__half));
  cudaMalloc(&d_output, n * n * sizeof(float));

  cudaMemcpy(d_input, input.data(), n * n * sizeof(__half),
             cudaMemcpyHostToDevice);

  dim3 block_size(4 * WARP_SIZE, 4);
  dim3 block_count((n / block_size.x) * (WARP_SIZE / WMMA_SIZE),
                   (n / block_size.y) / WMMA_SIZE);
  timer timer;
  WMMA_kernel<<<block_count, block_size>>>(d_input, d_output, n);
  cudaDeviceSynchronize();
  double time = timer.elapsed();
  std::cout << "Time: " << time << std::endl;

  cudaMemcpy(output.data(), d_output, n * n * sizeof(float),
             cudaMemcpyDeviceToHost);

  cudaFree(d_input);
  cudaFree(d_output);
}

void warmup_cutlass(const std::vector<__half> &matrix, std::size_t n) {
  throw std::runtime_error("CUTLASS warm-up not implemented");
}

cudaError_t CutlassGEMM(const cutlass::half_t *input, float *output,
                        const std::size_t n) {
  using RowMajor = cutlass::layout::RowMajor;
  using OpClassTesorOp = cutlass::arch::OpClassTensorOp;
  using Sm75 = cutlass::arch::Sm75;
  using CutlassGemm =
      cutlass::gemm::device::Gemm<cutlass::half_t, RowMajor,
                                  cutlass::half_t, RowMajor, float, RowMajor,
                                  float, OpClassTesorOp, Sm75>;

  CutlassGemm::Arguments args({cutlass::gemm::GemmCoord::Index(n), cutlass::gemm::GemmCoord::Index(n),
       cutlass::gemm::GemmCoord::Index(n)},
      {input, n}, {input, n}, {output, n},
                              {output, n}, {1, 0});

  CutlassGemm gemm_operator;
  cutlass::Status status = gemm_operator(args);
  return status == cutlass::Status::kSuccess ? cudaSuccess : cudaErrorUnknown;
}

void run_cutlass(const std::vector<__half> &input, std::vector<float> &output,
                 std::size_t n) {
  // throw std::runtime_error("CUTLASS method not implemented");

  cutlass::half_t *d_input;
  float *d_output;

  cudaMalloc(&d_input, n * n * sizeof(__half));
  cudaMalloc(&d_output, n * n * sizeof(float));

  cudaMemcpy(d_input, input.data(), n * n * sizeof(__half),
             cudaMemcpyHostToDevice);

  cudaError_t status = CutlassGEMM(d_input, d_output, n);
  cudaDeviceSynchronize();

  timer timer;
  status = CutlassGEMM(d_input, d_output, n);
  cudaDeviceSynchronize();
  double time = timer.elapsed();
  std::cout << "GEMM Time: " << time << std::endl;

  cudaMemcpy(output.data(), d_output, n * n * sizeof(float),
             cudaMemcpyDeviceToHost);

  cudaFree(d_input);
  cudaFree(d_output);
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

    std::size_t n = 1024;
    std::vector<__half> input(n * n, 0);
    make_input_matrix(input, n);
    // print_matrix<__half>(input, n);

    std::vector<float> openmp_result(n * n, 0);
    const double openmp_seconds = measure_seconds(
        [&]() { return run_openmp_reference(input, openmp_result, n); });
    std::cout << "OpenMP: " << format_time(openmp_seconds) << " sec\n";
    // print_matrix<float>(openmp_result, n);

    {
      RunResult mmgv1_res;
      try {
        mmgv1_res.result.resize(n * n, 0);
        mmgv1_res.seconds = measure_seconds([&]() {
          return run_matrix_mult_gpu_ver_1(input, mmgv1_res.result, n);
        });
        mmgv1_res.diff = max_abs_diff(openmp_result, mmgv1_res.result);
        mmgv1_res.success = true;
      } catch (const std::exception &ex) {
        std::cerr << "MMGV1 method failed: " << ex.what() << '\n';
      }
      print_report("MMGV1", mmgv1_res);
      // print_matrix<float>(mmgv1_res.result, n);
    }

    {
      RunResult mmgv2_res;
      try {
        mmgv2_res.result.resize(n * n, 0);
        mmgv2_res.seconds = measure_seconds([&]() {
          return run_matrix_mult_gpu_ver_2(input, mmgv2_res.result, n);
        });
        mmgv2_res.result.resize(n * n, 0);
        mmgv2_res.seconds = measure_seconds([&]() {
          return run_matrix_mult_gpu_ver_2(input, mmgv2_res.result, n);
        });
        mmgv2_res.diff = max_abs_diff(openmp_result, mmgv2_res.result);
        mmgv2_res.success = true;
      } catch (const std::exception &ex) {
        std::cerr << "MMGV2 method failed: " << ex.what() << '\n';
      }
      print_report("MMGV2", mmgv2_res);
      // print_matrix<float>(mmgv1_res.result, n);
    }
    {
      RunResult wmma_res;
      wmma_res.result.resize(n * n, 0);
      try {
        // warmup_wmma(input, n);
        wmma_res.seconds = measure_seconds(
            [&]() { return run_wmma(input, wmma_res.result, n); });
        wmma_res.diff = max_abs_diff(openmp_result, wmma_res.result);
        wmma_res.success = true;
      } catch (const std::exception &ex) {
        std::cerr << "WMMA method failed: " << ex.what() << '\n';
      }
      print_report("WMMA", wmma_res);
    }
    {
      RunResult cutlass_res;
      cutlass_res.result.resize(n * n, 0);
      try {
        cutlass_res.seconds = measure_seconds(
            [&]() { return run_cutlass(input, cutlass_res.result, n); });
        cutlass_res.diff = max_abs_diff(openmp_result, cutlass_res.result);
        cutlass_res.success = true;
      } catch (const std::exception &ex) {
        std::cerr << "CUTLASS method failed: " << ex.what() << '\n';
      }
      print_report("CUTLASS", cutlass_res);
    }

    return EXIT_SUCCESS;
  } catch (const std::exception &ex) {
    std::cerr << "Error: " << ex.what() << '\n';
  } catch (...) {
    std::cerr << "Unknown error\n";
  }

  return EXIT_FAILURE;
}
