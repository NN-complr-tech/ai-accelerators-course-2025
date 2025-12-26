#include <iostream>
#include "data_utils.h"

#ifndef ASCENDC_CPU_DEBUG
extern void matmul_custom_do(uint32_t block_dim, void* stream, uint8_t* matrix_a, uint8_t* matrix_b, uint8_t* matrix_c, uint8_t* tiling);
#include "acl/acl.h"
#else
#include "tikicpulib.h"
extern "C" __global__ __aicore__ void matmul_custom(GM_ADDR matrix_a, GM_ADDR matrix_b, GM_ADDR matrix_c, GM_ADDR tiling);
#endif


struct TileInfo {
  uint32_t n;
  uint32_t num_ai_cores;
  uint32_t sizeof_value;
  uint32_t plate_size; // размер одно блока 16 на 16 xDD

  uint32_t single_core_A; // такая длина должна быть выровнена по 16
  uint32_t single_core_B; // такая длина должна быть выровнена по 16
  uint32_t single_core_N;

  uint32_t base;
  uint32_t rows;
  uint32_t cols;
  
  uint32_t block_size; // размер блока на одно ai-core
  
  // uint32_t tile_count; // кол-во тайлов на одном ai-core
  // uint32_t tile_size; // размер блока, который пойдет в A1(B1)
  // uint32_t tile_last_size; // длина последнего tile

  // uint32_t plate_count; // кол-во блоков 16 на 16 на одном tile
  // uint32_t plate_count_last_tile; // кол-во блоков 16 на 16 на последнем tile
};

void GenerateTilingData(uint32_t n, TileInfo& tiling) {
   
  tiling.n = n;
  tiling.plate_size = 16;
  tiling.sizeof_value = sizeof(float);
  tiling.num_ai_cores = 8;
  tiling.base = 16;
  tiling.rows = 4;
  tiling.cols = 2;

  tiling.single_core_A = tiling.n / tiling.rows; // такая длина должна быть выровнена по 16
  tiling.single_core_B = tiling.n / tiling.cols; // такая длина должна быть выровнена по 16
  tiling.single_core_N = tiling.n / tiling.base;
  
  tiling.block_size = tiling.single_core_A * tiling.single_core_B;

  // uint32_t max_tile_block_size = 512; // L2 cache = 512KB -> 512 * 1024 / (16 * 16 * 4) = 512 * 1024 / 1024 = 512
  // uint32_t total_ai_cores_on_calculation = (tiling.n * tiling.n) / (tiling.plate_size * tiling.plate_size);
  // uint32_t max_ai_cores = 8;

  // if (total_ai_cores_on_calculation <= max_ai_cores)
  // {
  //   tiling.num_ai_cores = total_ai_cores_on_calculation;
  // }
  // else
  // {
  //   tiling.num_ai_cores = max_ai_cores;
  // }

  // tiling.block_size = (tiling.n * tiling.n) / (tiling.num_ai_cores);

  // uint32_t count_of_plate = tiling.block_size / tiling.plate_size; // кол-во блоков 16 на 16 на одном ai-core
  // tiling.tile_size = max_tile_block_size;
  // tiling.tile_count = (count_of_plate + tiling.tile_size - 1) / tiling.tile_size; // кол-во tileов на одном ai-core
  // tiling.tile_last_size = (count_of_plate % tiling.tile_size == 0) ? tiling.tile_size : count_of_plate % tiling.tile_size;

  // tiling.plate_count = tiling.tile_size / tiling.plate_size;
  // tiling.plate_count_last_tile = tiling.tile_last_size / tiling.plate_size;

}

int main()
{
  uint32_t n = 2048;

  std::size_t matrix_byte_size = n * n * sizeof(float);
  uint32_t block_dim = 8;

  TileInfo tiling;
  GenerateTilingData(n, tiling);

#ifdef ASCENDC_CPU_DEBUG
  uint8_t* a_mat = (uint8_t*)AscendC::GmAlloc(matrix_byte_size);
  uint8_t* b_mat = (uint8_t*)AscendC::GmAlloc(matrix_byte_size);
  uint8_t* c_mat = (uint8_t*)AscendC::GmAlloc(matrix_byte_size);

  ReadFile("./input/A.bin", matrix_byte_size, a_mat, matrix_byte_size);
  ReadFile("./input/B.bin", matrix_byte_size, b_mat, matrix_byte_size);

  ICPU_RUN_KF(matmul_custom, block_dim, a_mat, b_mat, c_mat, (uint8_t*)(&tiling));

  WriteFile("./output/output.bin", c_mat, matrix_byte_size);

  AscendC::GmFree(a_mat);
  AscendC::GmFree(b_mat);
  AscendC::GmFree(c_mat);
#else
  std::cout << "ELSE" << std::endl;
#endif


  return 0;
}