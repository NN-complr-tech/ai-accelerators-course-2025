#include <cutlass/gemm/device/gemm.h>
#include <utils.h>

void warmup_cutlass(const std::vector<__half> &matrix, std::size_t n) {
  throw std::runtime_error("CUTLASS warm-up not implemented");
}

cudaError_t CutlassGEMM(const cutlass::half_t *input, float *output,
                        const std::size_t n) {
  using RowMajor = cutlass::layout::RowMajor;
  using OpClassTesorOp = cutlass::arch::OpClassTensorOp;
  using Sm75 = cutlass::arch::Sm75;
  using CutlassGemm =
      cutlass::gemm::device::Gemm<cutlass::half_t, RowMajor, cutlass::half_t,
                                  RowMajor, float, RowMajor, float,
                                  OpClassTesorOp, Sm75>;

  CutlassGemm::Arguments args(
      {cutlass::gemm::GemmCoord::Index(n), cutlass::gemm::GemmCoord::Index(n),
       cutlass::gemm::GemmCoord::Index(n)},
      {input, n}, {input, n}, {output, n}, {output, n}, {1, 0});

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

  status = CutlassGEMM(d_input, d_output, n);
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