#include <iostream>

#include "data_utils.h"

struct TileInfo {
  uint32_t N;
  uint32_t num_of_ai_cores;
  uint32_t tile_length;
  uint32_t sizeof_type;
  uint32_t min_rows_per_core;
  uint32_t count_of_based_blocks;
  uint32_t count_of_cutted_blocks;
  uint32_t based_rows_per_block;
  uint32_t cutted_rows_per_block;
  uint32_t elems_per_tile;
  uint32_t tiles_per_row;
  uint32_t length_last_tile;
};

void GenerateTilingData(uint32_t n, TileInfo& tiling) {
  tiling.N = n;
  tiling.tile_length = 512;
  tiling.sizeof_type = sizeof(float);
  tiling.min_rows_per_core = 128;

  tiling.num_of_ai_cores =
      8 < ((n + tiling.min_rows_per_core - 1) / tiling.min_rows_per_core)
          ? 8
          : ((n + tiling.min_rows_per_core - 1) / tiling.min_rows_per_core);
  uint32_t remainder_rows = n % tiling.num_of_ai_cores;

  if (remainder_rows == 0) {
    tiling.count_of_based_blocks = tiling.num_of_ai_cores;
    tiling.count_of_cutted_blocks = 0;
    tiling.based_rows_per_block = n / tiling.num_of_ai_cores;
    tiling.cutted_rows_per_block = 0;
  } else {
    tiling.count_of_based_blocks = remainder_rows;
    tiling.count_of_cutted_blocks = tiling.num_of_ai_cores - remainder_rows;
    tiling.based_rows_per_block = n / tiling.num_of_ai_cores + 1;
    tiling.cutted_rows_per_block = n / tiling.num_of_ai_cores;
  }

  tiling.elems_per_tile = tiling.tile_length / tiling.sizeof_type;
  tiling.tiles_per_row =
      (n + tiling.elems_per_tile - 1) / tiling.elems_per_tile;
  tiling.length_last_tile = (n % tiling.elems_per_tile == 0)
                                ? tiling.elems_per_tile
                                : (n % tiling.elems_per_tile);
}

#ifndef ASCENDC_CPU_DEBUG

#include "acl/acl.h"
extern void exp_custom_do(
    uint32_t block_dim, void* stream, uint8_t* x, uint8_t* y, uint32_t N,
    uint32_t num_of_ai_cores, uint32_t tile_length, uint32_t sizeof_type,
    uint32_t min_rows_per_core, uint32_t count_of_based_blocks,
    uint32_t count_of_cutted_blocks, uint32_t based_rows_per_block,
    uint32_t cutted_rows_per_block, uint32_t elems_per_tile,
    uint32_t tiles_per_row, uint32_t length_last_tile);

#else

#include "tikicpulib.h"
extern "C" __global__ __aicore__ void exp_custom(
    GM_ADDR x, GM_ADDR y, uint32_t N, uint32_t num_of_ai_cores,
    uint32_t tile_length, uint32_t sizeof_type, uint32_t min_rows_per_core,
    uint32_t count_of_based_blocks, uint32_t count_of_cutted_blocks,
    uint32_t based_rows_per_block, uint32_t cutted_rows_per_block,
    uint32_t elems_per_tile, uint32_t tiles_per_row, uint32_t length_last_tile);
#endif

int main(int argc, char* argv[]) {
  uint32_t n = 2048;

  if (argc > 1) {
    int parsed = std::atoi(argv[1]);

    if (parsed > 0) {
      n = static_cast<uint32_t>(parsed);
    } else {
      std::cerr << "Invalid argument n, using default value: 2048\n";
    }
  }

  uint32_t min_rows_per_core = 128;
  uint32_t block_dim =
      (std::min)(8, int((n + min_rows_per_core - 1) / min_rows_per_core));
  std::size_t input_count_of_bytes = n * n * sizeof(float);
  std::size_t output_count_of_bytes = n * n * sizeof(float);

  TileInfo tiling;
  GenerateTilingData(n, tiling);

#ifdef ASCENDC_CPU_DEBUG

  std::cout << "SIZE IN MB: " << (float)input_count_of_bytes / (1024.0 * 1024.0)
            << std::endl;

  uint8_t* x = (uint8_t*)AscendC::GmAlloc(input_count_of_bytes);
  uint8_t* y = (uint8_t*)AscendC::GmAlloc(input_count_of_bytes);

  ReadFile("./input/input_x.bin", input_count_of_bytes, x,
           input_count_of_bytes);

  AscendC::SetKernelMode(KernelMode::AIV_MODE);

  ICPU_RUN_KF(exp_custom, block_dim, x, y, tiling.N, tiling.num_of_ai_cores,
              tiling.tile_length, tiling.sizeof_type, tiling.min_rows_per_core,
              tiling.count_of_based_blocks, tiling.count_of_cutted_blocks,
              tiling.based_rows_per_block, tiling.cutted_rows_per_block,
              tiling.elems_per_tile, tiling.tiles_per_row,
              tiling.length_last_tile);

  WriteFile("./output/output_y.bin", y, output_count_of_bytes);

  AscendC::GmFree(x);
  AscendC::GmFree(y);

#else

  CHECK_ACL(aclInit(nullptr));
  CHECK_ACL(aclrtSetDevice(0));
  aclrtStream stream = nullptr;
  CHECK_ACL(aclrtCreateStream(&stream));

  uint8_t *x_host, *y_host;
  uint8_t *x_device, *y_device;

  CHECK_ACL(aclrtMallocHost((void**)&x_host, input_count_of_bytes));
  CHECK_ACL(aclrtMallocHost((void**)&y_host, input_count_of_bytes));

  ReadFile("./input/input_x.bin", input_count_of_bytes, x_host,
           input_count_of_bytes);

  CHECK_ACL(aclrtMalloc((void**)&x_device, input_count_of_bytes,
                        ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMalloc((void**)&y_device, input_count_of_bytes,
                        ACL_MEM_MALLOC_HUGE_FIRST));

  CHECK_ACL(aclrtMemcpy(x_device, input_count_of_bytes, x_host,
                        input_count_of_bytes, ACL_MEMCPY_HOST_TO_DEVICE));

  exp_custom_do(block_dim, stream, x_device, y_device, tiling.N,
                tiling.num_of_ai_cores, tiling.tile_length, tiling.sizeof_type,
                tiling.min_rows_per_core, tiling.count_of_based_blocks,
                tiling.count_of_cutted_blocks, tiling.based_rows_per_block,
                tiling.cutted_rows_per_block, tiling.elems_per_tile,
                tiling.tiles_per_row, tiling.length_last_tile);

  CHECK_ACL(aclrtSynchronizeStream(stream));

  CHECK_ACL(aclrtMemcpy(y_host, output_count_of_bytes, y_device,
                        output_count_of_bytes, ACL_MEMCPY_DEVICE_TO_HOST));
  WriteFile("./output/output_y.bin", y_host, output_count_of_bytes);

  CHECK_ACL(aclrtFree(x_device));
  CHECK_ACL(aclrtFree(y_device));
  CHECK_ACL(aclrtFreeHost(x_host));
  CHECK_ACL(aclrtFreeHost(y_host));

  CHECK_ACL(aclrtDestroyStream(stream));
  CHECK_ACL(aclrtResetDevice(0));
  CHECK_ACL(aclFinalize());

#endif

  return 0;
}
