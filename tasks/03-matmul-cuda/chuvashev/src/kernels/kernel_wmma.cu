#include <mma.h>
#include <utils.h>

#define WARP_SIZE 32
#define WMMA_SIZE 16
using namespace nvcuda;

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
  WMMA_kernel<<<block_count, block_size>>>(d_input, d_output, n);
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  dim3 threads_per_block(THREADS_PER_BLOCK);
  dim3 blocks(n);
  softmax_kernel<<<blocks, threads_per_block>>>(d_output, n);
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  cudaMemcpy(output.data(), d_output, n * n * sizeof(float),
             cudaMemcpyDeviceToHost);

  cudaFree(d_input);
  cudaFree(d_output);
}