#include "kernel_operator.h"

class MatmulCustom
{
private:


public:

    __aicore__ inline void Init(GM_ADDR matrix_a, GM_ADDR matrix_b, GM_ADDR matrix_c)
    {
        AscendC::printf("Init()");
    }

    __aicore__ inline void Process()
    {
        AscendC::printf("Process()");
    }

};


extern "C" __global__ __aicore__ void matmul_custom(GM_ADDR matrix_a, GM_ADDR matrix_b, GM_ADDR matrix_c)
{
    MatmulCustom op;
    op.Init(matrix_a, matrix_b, matrix_c);
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
extern void matmul_custom_do(uint32_t block_dim, void* stream, uint8_t* matrix_a, uint8_t* matrix_b, uint8_t* matrix_c)
{
    matmul_custom<<<block_dim, nullptr, stream>>>(matrix_a, matrix_b, matrix_c);
}
#endif