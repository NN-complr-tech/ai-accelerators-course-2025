#include <iostream>
#include "data_utils.h"

#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
#else
#include "tikicpulib.h"
#endif



int main()
{
  uint32_t N = 1024;

  std::size_t matrix_byte_size = N * N * sizeof(float);


#ifdef ASCENDC_CPU_DEBUG
  uint8_t* a_mat = (uint8_t*)AscendC::GmAlloc(matrix_byte_size);
  uint8_t* b_mat = (uint8_t*)AscendC::GmAlloc(matrix_byte_size);
  uint8_t* c_mat = (uint8_t*)AscendC::GmAlloc(matrix_byte_size);

  ReadFile("./input/A.bin", matrix_byte_size, a_mat, matrix_byte_size);
  ReadFile("./input/B.bin", matrix_byte_size, b_mat, matrix_byte_size);

  //ICPU_RUN_KF();

  WriteFile("./output/output.bin", c_mat, matrix_byte_size);

  AscendC::GmFree(a_mat);
  AscendC::GmFree(b_mat);
  AscendC::GmFree(c_mat);
#else
  std::cout << "ELSE" << std::endl;
#endif


  return 0;
}