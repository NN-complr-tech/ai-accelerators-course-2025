#include "kernel_operator.h"

class KernelSoftmax {
 private:
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

  uint32_t global_offset = 0;
  uint32_t count_of_rows = 0;

  AscendC::TPipe* pipe;
  AscendC::TQue<AscendC::TPosition::VECIN, 1> in_queue_x;
  AscendC::TQue<AscendC::TPosition::VECOUT, 1> out_queue_y;

  AscendC::TBuf<AscendC::TPosition::VECCALC> buffer_for_sum;
  AscendC::TBuf<AscendC::TPosition::VECCALC> buffer_for_exp;
  AscendC::TBuf<AscendC::TPosition::VECCALC> buffer_for_reduce;
  AscendC::TBuf<AscendC::TPosition::VECCALC> buffer_for_div;

  AscendC::GlobalTensor<float> x_global;
  AscendC::GlobalTensor<float> y_global;

 public:
  __aicore__ inline KernelSoftmax(
      uint32_t _N, uint32_t _num_of_ai_cores, uint32_t _tile_length,
      uint32_t _sizeof_type, uint32_t _min_rows_per_core,
      uint32_t _count_of_based_blocks, uint32_t _count_of_cutted_blocks,
      uint32_t _based_rows_per_block, uint32_t _cutted_rows_per_block,
      uint32_t _elems_per_tile, uint32_t _tiles_per_row,
      uint32_t _length_last_tile)
      : N(_N),
        num_of_ai_cores(_num_of_ai_cores),
        tile_length(_tile_length),
        sizeof_type(_sizeof_type),
        min_rows_per_core(_min_rows_per_core),
        count_of_based_blocks(_count_of_based_blocks),
        count_of_cutted_blocks(_count_of_cutted_blocks),
        based_rows_per_block(_based_rows_per_block),
        cutted_rows_per_block(_cutted_rows_per_block),
        elems_per_tile(_elems_per_tile),
        tiles_per_row(_tiles_per_row),
        length_last_tile(_length_last_tile) {}

  __aicore__ inline void Init(AscendC::TPipe* p, GM_ADDR x, GM_ADDR y) {
    pipe = p;
    uint32_t block_idx = AscendC::GetBlockIdx();

    if (block_idx < count_of_based_blocks) {
      global_offset = block_idx * based_rows_per_block * N;
      count_of_rows = based_rows_per_block;
    } else {
      global_offset = count_of_based_blocks * based_rows_per_block * N;
      global_offset +=
          (block_idx - count_of_based_blocks) * cutted_rows_per_block * N;
      count_of_rows = cutted_rows_per_block;
    }

    x_global.SetGlobalBuffer((__gm__ float*)x + global_offset,
                             count_of_rows * N * sizeof(float));
    y_global.SetGlobalBuffer((__gm__ float*)y + global_offset,
                             count_of_rows * N * sizeof(float));

    pipe->InitBuffer(in_queue_x, 1, tile_length);
    pipe->InitBuffer(out_queue_y, 1, tile_length);

    pipe->InitBuffer(buffer_for_sum, elems_per_tile * sizeof(float));
    pipe->InitBuffer(buffer_for_exp, elems_per_tile * sizeof(float));
    pipe->InitBuffer(buffer_for_div, elems_per_tile * sizeof(float));
    pipe->InitBuffer(buffer_for_reduce, sizeof(float));
  }

  __aicore__ inline void CopyIn(uint32_t r, uint32_t t) {
    AscendC::LocalTensor<float> x_local = in_queue_x.AllocTensor<float>();
    uint32_t offset = r * N + t * elems_per_tile;
    uint32_t elems =
        (t == tiles_per_row - 1) ? length_last_tile : elems_per_tile;

    AscendC::DataCopy(x_local, x_global[offset], elems);
    in_queue_x.EnQue(x_local);
  }

  __aicore__ inline void ComputeExpsAndSum(uint32_t r, uint32_t t,
                                           AscendC::LocalTensor<float>& exps,
                                           AscendC::LocalTensor<float>& sums) {
    uint32_t elems =
        (t == tiles_per_row - 1) ? length_last_tile : elems_per_tile;

    AscendC::LocalTensor<float> x_local = in_queue_x.DeQue<float>();

    AscendC::Exp(exps, x_local, elems);
    AscendC::Add(sums, exps, sums, elems);

    in_queue_x.FreeTensor(x_local);
  }

  __aicore__ inline void DivideOnExps(uint32_t r, uint32_t t) {
    uint32_t elems =
        (t == tiles_per_row - 1) ? length_last_tile : elems_per_tile;

    AscendC::LocalTensor<float> x_local = in_queue_x.DeQue<float>();
    AscendC::LocalTensor<float> y_local = out_queue_y.AllocTensor<float>();

    AscendC::LocalTensor<float> div = buffer_for_div.Get<float>();

    AscendC::Exp(x_local, x_local, elems);
    AscendC::Div(y_local, x_local, div, elems);

    in_queue_x.FreeTensor(x_local);
    out_queue_y.EnQue(y_local);
  }

  __aicore__ inline void CopyOut(uint32_t r, uint32_t t) {
    uint32_t offset = r * N + t * elems_per_tile;
    uint32_t elems =
        (t == tiles_per_row - 1) ? length_last_tile : elems_per_tile;

    AscendC::LocalTensor<float> y_local = out_queue_y.DeQue<float>();

    AscendC::DataCopy(y_global[offset], y_local, elems);

    out_queue_y.FreeTensor(y_local);
  }

  __aicore__ inline void Process() {
    AscendC::LocalTensor<float> sum_tensor =
        buffer_for_sum.Get<float>();  // тензор для хранения суммы exp по всем
                                      // тайлам в строке
    AscendC::LocalTensor<float> exp_tensor =
        buffer_for_exp.Get<float>();  // тензор для хранения exp текущего тайла
    AscendC::LocalTensor<float> reduce_scalar =
        buffer_for_reduce.Get<float>();  // хранит значение редуцированной суммы
    AscendC::LocalTensor<float> div =
        buffer_for_div.Get<float>();  // хранит элементы, на которые будем
                                      // делить тайлы в строке

    for (uint32_t r = 0; r < count_of_rows; ++r) {
      AscendC::Duplicate(sum_tensor, 0.0f, elems_per_tile);

      for (uint32_t t = 0; t < tiles_per_row; ++t) {
        CopyIn(r, t);
        ComputeExpsAndSum(r, t, exp_tensor, sum_tensor);
      }

      const uint32_t shape[] = {1, elems_per_tile};
      AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR>(
          reduce_scalar, sum_tensor, shape, true);

      float value = reduce_scalar.GetValue(0);
      AscendC::Duplicate(div, value, elems_per_tile);

      for (uint32_t t = 0; t < tiles_per_row; ++t) {
        CopyIn(r, t);
        DivideOnExps(r, t);
        CopyOut(r, t);
      }
    }
  }
};

extern "C" __global__ __aicore__ void exp_custom(
    GM_ADDR x, GM_ADDR y,

    uint32_t N, uint32_t num_of_ai_cores, uint32_t tile_length,
    uint32_t sizeof_type, uint32_t min_rows_per_core,
    uint32_t count_of_based_blocks, uint32_t count_of_cutted_blocks,
    uint32_t based_rows_per_block, uint32_t cutted_rows_per_block,
    uint32_t elems_per_tile, uint32_t tiles_per_row,
    uint32_t length_last_tile) {
  KernelSoftmax op(
      N, num_of_ai_cores, tile_length, sizeof_type, min_rows_per_core,
      count_of_based_blocks, count_of_cutted_blocks, based_rows_per_block,
      cutted_rows_per_block, elems_per_tile, tiles_per_row, length_last_tile);

  AscendC::TPipe pipe;
  op.Init(&pipe, x, y);
  op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
extern void exp_custom_do(
    uint32_t block_dim, void* stream, uint8_t* x, uint8_t* y, uint32_t N,
    uint32_t num_of_ai_cores, uint32_t tile_length, uint32_t sizeof_type,
    uint32_t min_rows_per_core, uint32_t count_of_based_blocks,
    uint32_t count_of_cutted_blocks, uint32_t based_rows_per_block,
    uint32_t cutted_rows_per_block, uint32_t elems_per_tile,
    uint32_t tiles_per_row, uint32_t length_last_tile) {
  exp_custom<<<block_dim, nullptr, stream>>>(
      x, y, N, num_of_ai_cores, tile_length, sizeof_type, min_rows_per_core,
      count_of_based_blocks, count_of_cutted_blocks, based_rows_per_block,
      cutted_rows_per_block, elems_per_tile, tiles_per_row, length_last_tile);
}
#endif
