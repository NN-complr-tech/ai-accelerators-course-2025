#include <utils.h>

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
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  dim3 threads_per_block(THREADS_PER_BLOCK);
  dim3 blocks(n);
  softmax_kernel<<<blocks, threads_per_block>>>(d_output, n);
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  cudaMemcpy(output.data(), d_output, sizeof(float) * n * n,
             cudaMemcpyDeviceToHost);

  cudaFree(d_input);
  cudaFree(d_output);
}