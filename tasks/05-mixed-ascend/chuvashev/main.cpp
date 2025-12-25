#include <iostream>
#include "data_utils.h"

#ifndef ASCENDC_CPU_DEBUG
extern void matmul_custom_do(uint32_t block_dim, void* stream, uint8_t* matrix_a, uint8_t* matrix_b, uint8_t* matrix_c);
#include "acl/acl.h"
#else
#include "tikicpulib.h"
extern "C" __global__ __aicore__ void matmul_custom(GM_ADDR matrix_a, GM_ADDR matrix_b, GM_ADDR matrix_c, uint32_t n);
#endif


int main()
{
  uint32_t N = 32;

  std::size_t matrix_byte_size = N * N * sizeof(float);
  uint32_t block_dim = 8;

#ifdef ASCENDC_CPU_DEBUG
  uint8_t* a_mat = (uint8_t*)AscendC::GmAlloc(matrix_byte_size);
  uint8_t* b_mat = (uint8_t*)AscendC::GmAlloc(matrix_byte_size);
  uint8_t* c_mat = (uint8_t*)AscendC::GmAlloc(matrix_byte_size);

  ReadFile("./input/A.bin", matrix_byte_size, a_mat, matrix_byte_size);
  ReadFile("./input/B.bin", matrix_byte_size, b_mat, matrix_byte_size);

  ICPU_RUN_KF(matmul_custom, block_dim, a_mat, b_mat, c_mat, N);

  WriteFile("./output/output.bin", c_mat, matrix_byte_size);

  AscendC::GmFree(a_mat);
  AscendC::GmFree(b_mat);
  AscendC::GmFree(c_mat);
#else
  std::cout << "ELSE" << std::endl;
#endif


  return 0;
}