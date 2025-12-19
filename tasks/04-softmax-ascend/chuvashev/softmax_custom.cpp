#include "kernel_operator.h"

class KernelSoftmax
{
private:

    uint32_t global_offset = 0;
    uint32_t count_of_rows = 0;

    AscendC::TPipe* pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> in_queue_x;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> out_queue_y;
    AscendC::TBuf<AscendC::TPosition::VECCALC> buffer_for_sum;
    AscendC::TBuf<AscendC::TPosition::VECCALC> buffer_for_div;
    AscendC::GlobalTensor<float> x_global_tensor;
    AscendC::GlobalTensor<float> y_global_tensor;

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

    void GenerateTilingData(uint32_t n)
    {
        N = n;

        tile_length = 512;
        sizeof_type = sizeof(float);
        min_rows_per_core = 128;
        
        num_of_ai_cores = 8 < ((n + min_rows_per_core - 1) / min_rows_per_core) 
                                ? 8 
                                : ((n + min_rows_per_core - 1) / min_rows_per_core);

        uint32_t remainder_rows = n % num_of_ai_cores;

        if (remainder_rows == 0)
        {
            count_of_based_blocks = num_of_ai_cores;
            count_of_cutted_blocks = 0;
            based_rows_per_block = n / num_of_ai_cores;
            cutted_rows_per_block = 0;
        }
        else
        {
            count_of_based_blocks = remainder_rows;
            count_of_cutted_blocks = num_of_ai_cores - remainder_rows;
            based_rows_per_block = n / num_of_ai_cores + 1;
            cutted_rows_per_block = n / num_of_ai_cores;
        }

        elems_per_tile = tile_length / sizeof_type;
        tiles_per_row = (n + elems_per_tile - 1) / elems_per_tile;
        length_last_tile = (n % elems_per_tile == 0) ? elems_per_tile : (n % elems_per_tile);
    }

public:
    __aicore__ inline KernelSoftmax(uint32_t n) 
    {
        GenerateTilingData(n);
    }

    __aicore__ inline void Init(AscendC::TPipe* pipe_in, GM_ADDR x, GM_ADDR y)
    {
        uint32_t block_idx = AscendC::GetBlockIdx();

        if (block_idx < count_of_based_blocks)
        {
            global_offset = block_idx * based_rows_per_block * N;
            count_of_rows = based_rows_per_block;
        }
        else
        {
            global_offset = count_of_based_blocks * based_rows_per_block * N;
            global_offset += (block_idx - count_of_based_blocks) * cutted_rows_per_block * N;
            count_of_rows = cutted_rows_per_block;
        }

        x_global_tensor.SetGlobalBuffer((__gm__ float*)x + global_offset, 
                                       count_of_rows * N * sizeof(float));
        y_global_tensor.SetGlobalBuffer((__gm__ float*)y + global_offset, 
                                       count_of_rows * N * sizeof(float));

        pipe = pipe_in;
        pipe->InitBuffer(in_queue_x, 1, tile_length);
        pipe->InitBuffer(out_queue_y, 1, tile_length);
        pipe->InitBuffer(buffer_for_sum, elems_per_tile * sizeof(float));
        pipe->InitBuffer(buffer_for_div, elems_per_tile * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t row = 0; row < count_of_rows; ++row)
        {
            auto sum = buffer_for_sum.Get<float>();
            AscendC::Duplicate(sum, 0.0f, 1);

            for (uint32_t tile = 0; tile < tiles_per_row; ++tile)
            {
                uint32_t progress = row * N + tile * elems_per_tile;
                uint32_t elems = (tile == tiles_per_row - 1)
                                ? length_last_tile
                                : elems_per_tile;

                AscendC::LocalTensor<float> x = in_queue_x.AllocTensor<float>();
                AscendC::LocalTensor<float> y = out_queue_y.AllocTensor<float>();

                AscendC::DataCopy(x, x_global_tensor[progress], elems);
                AscendC::Exp(y, x, elems);

                auto tmp = buffer_for_div.Get<float>();
                uint32_t shape[] = {1, elems};

                AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR>(
                    tmp, y, shape, true
                );

                AscendC::Add(sum, sum, tmp, 1);

                in_queue_x.FreeTensor(x);
                out_queue_y.FreeTensor(y);
            }

            auto div = buffer_for_div.Get<float>();
            AscendC::Duplicate(div, sum.GetValue(0), elems_per_tile);

            for (uint32_t tile = 0; tile < tiles_per_row; ++tile)
            {
                uint32_t progress = row * N + tile * elems_per_tile;
                uint32_t elems = (tile == tiles_per_row - 1) 
                                ? length_last_tile 
                                : elems_per_tile;

                AscendC::LocalTensor<float> x = in_queue_x.AllocTensor<float>();
                AscendC::LocalTensor<float> y = out_queue_y.AllocTensor<float>();

                AscendC::DataCopy(x, x_global_tensor[progress], elems);
                AscendC::Exp(y, x, elems);
                in_queue_x.FreeTensor(x);

                AscendC::Div(y, y, div, elems);
                AscendC::DataCopy(y_global_tensor[progress], y, elems);
                out_queue_y.FreeTensor(y);
            }
        }
    }
};

extern "C" __global__ __aicore__ void exp_custom(GM_ADDR x, GM_ADDR y, uint32_t n)
{
    AscendC::TPipe pipe;
    KernelSoftmax op(n);
    op.Init(&pipe, x, y);
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
void exp_custom_do(uint32_t block_dim, void* stream, uint8_t* x, uint8_t* y, uint32_t n)
{
    exp_custom<<<block_dim, nullptr, stream>>>(x, y, n);
}
#endif