#include <iostream>
#include "data_utils.h"

#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
#else
#include "tikicpulib.h"
#endif



int main()
{

  uint32_t M = ;
  uint32_t K = ;
  uint32_t N = ;

  uint32_t matrix_a_byte_size = M * K * sizeof(float);
  uitn32_t matrix_b_byte_size = N * K * sizeof(float);
  uitn32_t matrix_c_byte_size = M * N * sizeof(float);

#ifdef ASCENDC_CPU_DEBUG

  uint8_t* a_mat = (uint8_t*)AscendC::GmAlloc(matrix_a_byte_size),
  uint8_t* b_mat = (uint8_t*)AscendC::GmAlloc(matrix_b_byte_size);
  uint8_t* c_mat = (uint8_t*)AscendC::GmAlloc(matrix_c_byte_size);

  ReadFile("./input/A.bin", matrix_a_byte_size, a_mat, matrix_a_byte_size);
  ReadFile("./input/B.bin", matrix_b_byte_size, b_mat, matrix_b_byte_size);

  ICPU_RUN_KF();

  WriteFile("./output/C.bin", matrix_c_byte_size, c_mat, matrix_c_byte_size);

#else

#endif

  std::cout << "uioadsda" << std::endl;

  return 0;
}