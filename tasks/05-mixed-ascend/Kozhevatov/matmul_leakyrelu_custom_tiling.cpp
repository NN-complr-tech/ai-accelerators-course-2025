/**
 * @file matmul_leakyrelu_custom_tiling.cpp
 * Tiling configuration с параметрами из config.h
 *
 * Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include <cassert>
#include <fstream>
#include <iostream>

#include "config.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

using namespace matmul_tiling;
using namespace std;

/**
 * @brief  Generate matmul tiling.
 * @param  socVersion: Platform socversion.
 * @param  tilingBuf data buffer.
 */
void GenerateTiling(const char* socVersion, uint8_t* tilingBuf) {
  // Параметры матрицы из config.h
  int M = MATRIX_SIZE;
  int N = MATRIX_SIZE;
  int K = MATRIX_SIZE;

  TPosition leftPosition = TPosition::GM;
  CubeFormat leftFormat = CubeFormat::ND;
  DataType leftDtype = DataType::DT_FLOAT16;
  bool isTransA = false;

  TPosition rightPosition = TPosition::GM;
  CubeFormat rightFormat = CubeFormat::ND;
  DataType rightDtype = DataType::DT_FLOAT16;
  bool isTransB = false;

  TPosition resultPosition = TPosition::GM;
  CubeFormat resultFormat = CubeFormat::ND;
  DataType resultDtype = DataType::DT_FLOAT;

  TPosition biasPosition = TPosition::GM;
  CubeFormat biasFormat = CubeFormat::ND;
  DataType biasDtype = DataType::DT_FLOAT;
  bool isBias = true;

  // Используем константы из config.h
  int usedCoreNum = MATMUL_CORE_NUM;
  int baseM = MATRIX_SIZE;
  int baseN = MATRIX_SIZE / usedCoreNum;

  optiling::TCubeTiling tilingData;
  auto ascendcPlatform =
      platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);
  MultiCoreMatmulTiling tilingApi(*ascendcPlatform);

  tilingApi.SetDim(usedCoreNum);
  tilingApi.SetAType(leftPosition, leftFormat, leftDtype, isTransA);
  tilingApi.SetBType(rightPosition, rightFormat, rightDtype, isTransB);
  tilingApi.SetCType(resultPosition, resultFormat, resultDtype);
  tilingApi.SetBiasType(biasPosition, biasFormat, biasDtype);

  tilingApi.SetOrgShape(M, N, K);
  tilingApi.SetShape(M, N, K);
  tilingApi.SetBias(isBias);
  tilingApi.SetTraverse(MatrixTraverse::FIRSTM);
  tilingApi.SetFixSplit(baseM, baseN, -1);
  tilingApi.SetBufferSpace(-1, -1, -1);

  int64_t res = tilingApi.GetTiling(tilingData);
  tilingData.set_stepM(1);
  tilingData.set_stepN(1);

  if (res == -1) {
    std::cout << "gen tiling failed" << std::endl;
  }

  uint32_t tcubeTilingSize = tilingData.GetDataSize();
  tilingData.SaveToBuffer(tilingBuf, tcubeTilingSize);

  std::cout << "✓ Tiling generated: M=" << M << ", N=" << N << ", K=" << K
            << ", cores=" << usedCoreNum << std::endl;
}