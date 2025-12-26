#include "data_utils.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
#include "aclrtlaunch_matmul_custom.h"
#else
#include "tikicpulib.h"
extern "C" void matmul_custom(uint8_t *a, uint8_t *b, uint8_t *c,
                              uint8_t *workspace, uint8_t *tiling);
#endif
extern void GenerateTiling(const char *socVersion, uint8_t *tilingBuf,
                           uint32_t n);

#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
extern void exp_custom_do(uint32_t block_dim, void *stream, uint8_t *x,
                          uint8_t *y, uint8_t *tiling);
#else
#include "tikicpulib.h"
extern "C" __global__ __aicore__ void exp_custom(GM_ADDR x, GM_ADDR y,
                                                 GM_ADDR tiling);
#endif

struct TileInfo {
  uint32_t N;
  uint32_t M;
  uint32_t num_of_ai_cores;
  uint32_t tile_length;  // длина одного тайла (В байтах)
  uint32_t sizeof_type;  // размер 1 элемента (В байтах)
  uint32_t count_of_based_blocks;
  uint32_t count_of_cutted_blocks;
  uint32_t based_rows_per_block;
  uint32_t cutted_rows_per_block;
  uint32_t elems_per_tile;  // кол-во элементов на каждый тайл (НЕ в байтах)
  uint32_t tiles_per_row;
  uint32_t
      length_last_tile;  // кол-во элементов на последнем тайле (НЕ в байтах)
  uint32_t length_last_tile_align;  // кол-во элементов на последнем тайле с
                                    // выравниванием по 32 байтам (НЕ в байтах)
  uint32_t buffer_num;
};

void GenerateTilingData(uint32_t n, TileInfo &tiling) {
  tiling.N = n;
  tiling.buffer_num = 2;

  tiling.num_of_ai_cores = 1;

  tiling.tile_length =
      1024 / tiling.buffer_num;  // учитываем, что DoubleBuffering
  tiling.sizeof_type = sizeof(float);

  std::size_t bytes = n * tiling.sizeof_type;
  std::size_t size_of_vec = 32;
  tiling.M = tiling.N;

  if (bytes % size_of_vec != 0) {
    std::size_t cut_bytes = bytes % size_of_vec;
    std::size_t additional_bytes = size_of_vec - cut_bytes;

    tiling.M +=
        additional_bytes / tiling.sizeof_type;  // выровняли по 32 байтам
  }

  uint32_t remainder_rows =
      n % tiling.num_of_ai_cores;  // кол-во строк, которые не останутся не
                                   // обработанными

  if (remainder_rows == 0) {  // каждый ai-core получает одинковое число строк
    tiling.count_of_based_blocks = tiling.num_of_ai_cores;
    tiling.count_of_cutted_blocks = 0;
    tiling.based_rows_per_block = n / tiling.num_of_ai_cores;
    tiling.cutted_rows_per_block = 0;
  } else {  // все блоки получают n / tiling.num_of_ai_cores строк, а также
            // remainder_rows блоков получают дополнительно по 1 строке
    tiling.count_of_based_blocks = remainder_rows;
    tiling.count_of_cutted_blocks = tiling.num_of_ai_cores - remainder_rows;
    tiling.based_rows_per_block = n / tiling.num_of_ai_cores + 1;
    tiling.cutted_rows_per_block = n / tiling.num_of_ai_cores;
  }

  if (tiling.M == tiling.N)  // данные выровнены по 32 байтам (нужно учесть, что
                             // DobuleBuffering)
  {
    tiling.elems_per_tile = tiling.tile_length / tiling.sizeof_type;
    tiling.tiles_per_row =
        (tiling.N + tiling.elems_per_tile - 1) /
        tiling.elems_per_tile;  // тут используется длина не alignутой строки
    tiling.length_last_tile = (tiling.N % tiling.elems_per_tile == 0)
                                  ? tiling.elems_per_tile
                                  : (tiling.N % tiling.elems_per_tile);
    tiling.length_last_tile_align = tiling.length_last_tile;
  } else {
    tiling.elems_per_tile = tiling.tile_length / tiling.sizeof_type;
    tiling.tiles_per_row =
        (tiling.N + tiling.elems_per_tile - 1) /
        tiling.elems_per_tile;  // тут используется длина не alignутой строки
    tiling.length_last_tile = (tiling.N % tiling.elems_per_tile == 0)
                                  ? tiling.elems_per_tile
                                  : (tiling.N % tiling.elems_per_tile);
    tiling.length_last_tile_align =
        (tiling.M - tiling.N) + tiling.length_last_tile % tiling.elems_per_tile;
  }
}

int32_t main(int32_t argc, char *argv[]) {
  uint32_t n = 512;
  if (argc > 1) {
    int parsed = std::atoi(argv[1]);

    if (parsed > 0) {
      n = static_cast<uint32_t>(parsed);
    } else {
      std::cerr << "Invalid argument n, using default value: 512\n";
    }
  }
  const char *socVersion = SOC_VERSION;
  auto ascendcPlatform =
      platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);
  size_t aFileSize = n * n * sizeof(uint16_t);  // uint16_t represent half
  size_t bFileSize = n * n * sizeof(uint16_t);  // uint16_t represent half
  size_t cFileSize = n * n * sizeof(float);
  // matmul TCubeTiling + localMemorySize
  size_t tilingFileSize = sizeof(TCubeTiling) + sizeof(uint64_t);
  size_t userWorkspaceSize = 0;
  size_t systemWorkspaceSize =
      static_cast<size_t>(ascendcPlatform->GetLibApiWorkSpaceSize());
  size_t workspaceSize = userWorkspaceSize + systemWorkspaceSize;
  uint8_t *tilingBuf = (uint8_t *)malloc(tilingFileSize);
  GenerateTiling(socVersion, tilingBuf, n);
#ifdef CUSTOM_ASCEND310P
  uint32_t blockDim = 2;
#else
  uint32_t blockDim = 1;
#endif

  TileInfo tiling_softmax;
  GenerateTilingData(n, tiling_softmax);

#ifdef ASCENDC_CPU_DEBUG
  uint8_t *a = (uint8_t *)AscendC::GmAlloc(aFileSize);
  uint8_t *b = (uint8_t *)AscendC::GmAlloc(bFileSize);
  uint8_t *c = (uint8_t *)AscendC::GmAlloc(cFileSize);
  uint8_t *output = (uint8_t *)AscendC::GmAlloc(cFileSize);
  uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceSize);
  uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingFileSize);

  ReadFile("./input/A.bin", aFileSize, a, aFileSize);
  ReadFile("./input/B.bin", bFileSize, b, bFileSize);
  memcpy_s(tiling, tilingFileSize, tilingBuf, tilingFileSize);

  ICPU_RUN_KF(matmul_custom, blockDim, a, b, c, workspace, tiling);

  WriteFile("./output/output_mult.bin", c, cFileSize);

  ICPU_RUN_KF(exp_custom, tiling_softmax.num_of_ai_cores, c, output,
              (uint8_t *)(&tiling_softmax));

  WriteFile("./output/output.bin", output, cFileSize);

  AscendC::GmFree((void *)a);
  AscendC::GmFree((void *)b);
  AscendC::GmFree((void *)c);
  AscendC::GmFree((void *)output);
  AscendC::GmFree((void *)workspace);
  AscendC::GmFree((void *)tiling);
#else
  CHECK_ACL(aclInit(nullptr));
  int32_t deviceId = 0;
  CHECK_ACL(aclrtSetDevice(deviceId));
  aclrtStream stream = nullptr;
  CHECK_ACL(aclrtCreateStream(&stream));

  uint8_t *aHost;
  uint8_t *aDevice;
  CHECK_ACL(aclrtMallocHost((void **)(&aHost), aFileSize));
  CHECK_ACL(
      aclrtMalloc((void **)&aDevice, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
  ReadFile("./input/A.bin", aFileSize, aHost, aFileSize);
  CHECK_ACL(aclrtMemcpy(aDevice, aFileSize, aHost, aFileSize,
                        ACL_MEMCPY_HOST_TO_DEVICE));

  uint8_t *bHost;
  uint8_t *bDevice;
  CHECK_ACL(aclrtMallocHost((void **)(&bHost), bFileSize));
  CHECK_ACL(
      aclrtMalloc((void **)&bDevice, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
  ReadFile("./input/B.bin", bFileSize, bHost, bFileSize);
  CHECK_ACL(aclrtMemcpy(bDevice, bFileSize, bHost, bFileSize,
                        ACL_MEMCPY_HOST_TO_DEVICE));

  uint8_t *workspaceDevice;
  CHECK_ACL(aclrtMalloc((void **)&workspaceDevice, workspaceSize,
                        ACL_MEM_MALLOC_HUGE_FIRST));

  uint8_t *tilingHost;
  uint8_t *tilingDevice;
  CHECK_ACL(aclrtMallocHost((void **)(&tilingHost), tilingFileSize));
  CHECK_ACL(aclrtMalloc((void **)&tilingDevice, tilingFileSize,
                        ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMemcpy(tilingHost, tilingFileSize, tilingBuf, tilingFileSize,
                        ACL_MEMCPY_HOST_TO_HOST));
  CHECK_ACL(aclrtMemcpy(tilingDevice, tilingFileSize, tilingHost,
                        tilingFileSize, ACL_MEMCPY_HOST_TO_DEVICE));

  uint8_t *cDevice;
  CHECK_ACL(
      aclrtMalloc((void **)&cDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST));

  ACLRT_LAUNCH_KERNEL(matmul_custom)
  (blockDim, stream, aDevice, bDevice, cDevice, workspaceDevice, tilingDevice);
  CHECK_ACL(aclrtSynchronizeStream(stream));

  CHECK_ACL(aclrtFree(aDevice));
  CHECK_ACL(aclrtFreeHost(aHost));
  CHECK_ACL(aclrtFree(bDevice));
  CHECK_ACL(aclrtFreeHost(bHost));
  CHECK_ACL(aclrtFree(workspaceDevice));
  CHECK_ACL(aclrtFree(tilingDevice));
  CHECK_ACL(aclrtFreeHost(tilingHost));

  uint8_t *tiling_device, *output_device;
  uint8_t *output_host;

  CHECK_ACL(aclrtMallocHost((void **)(&output_host), cFileSize));
  CHECK_ACL(aclrtMalloc((void **)&output_device, cFileSize,
                        ACL_MEM_MALLOC_HUGE_FIRST));

  CHECK_ACL(aclrtMalloc((void **)&tiling_device, sizeof(TileInfo),
                        ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMemcpy(tiling_device, sizeof(TileInfo),
                        (uint8_t *)(&tiling_softmax), sizeof(TileInfo),
                        ACL_MEMCPY_HOST_TO_DEVICE));

  exp_custom_do(tiling_softmax.num_of_ai_cores, stream, cDevice, output_device,
                tiling_device);
  CHECK_ACL(aclrtSynchronizeStream(stream));

  CHECK_ACL(aclrtMemcpy(output_host, cFileSize, output_device, cFileSize,
                        ACL_MEMCPY_DEVICE_TO_HOST));
  WriteFile("./output/output.bin", output_host, cFileSize);

  CHECK_ACL(aclrtFree(cDevice));
  CHECK_ACL(aclrtFree(output_device));
  CHECK_ACL(aclrtFreeHost(output_host));
  CHECK_ACL(aclrtFree(tiling_device));

  CHECK_ACL(aclrtDestroyStream(stream));
  CHECK_ACL(aclrtResetDevice(deviceId));
  CHECK_ACL(aclFinalize());
#endif
  free(tilingBuf);
  return 0;
}