#include "kernel_operator.h"

class MatmulCustom
{
private:

    AscendC::GlobalTensor<float> global_matrix_a;
    AscendC::GlobalTensor<float> global_matrix_b;
    AscendC::GlobalTensor<float> global_matrix_c;

    AscendC::TPipe *pipe;
    AscendC::TQue<AscendC::TPosition::A1, 1> in_queue_A1;
    AscendC::TQue<AscendC::TPosition::B1, 1> in_queue_B1;

    AscendC::TQue<AscendC::TPosition::A2, 1> in_queue_A2;

    uint32_t n;
    uint32_t blocks;


    __aicore__ void CopyND2NZ(AscendC::LocalTensor<float>& dst, AscendC::GlobalTensor<float>& src, const uint16_t heigth, const uint16_t width)
    {
        // формат NZ
        // Z  /Z  /Z
        // Z / Z / Z
        // Z/  Z/  Z
        
        // изначальные данные у нас в row-major порядке
        // обрабатываем блоками 16 на 16
        for (uint32_t i = 0; i < width / 16; ++i)
        {
            uint32_t src_offset = i * 16; // берем элементы по строчно блоками из 16
            uint32_t dst_offset = i * 16 * heigth; // кладем их в ячейки также по строчно но со смещением в высоту * 16 (из-за NZ формата)
            
            // https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/API/ascendcopapi/atlasascendc_api_07_00127.html
            // 1-ый параметр - кол-во передаваемых матриц (сколько раз будет повтор DataCopy)
            // 2-ой параметр - расстояние между матрицами в исходном тензоре (1, так как блоки идут друг за другом)
            // 3-ий параметр - расстояние между матрицами в целевом тензоре (width / 16 - 1 отступ до следующего элемента)
            AscendC::DataCopy(dst[dst_offset], src[src_offset], { heigth, 1, uint16_t(width / 16 - 1), 0 });
        }

    }

    __aicore__ inline void CopyIn()
    {
        // выделили тензор для хранения БОЛЬШОЙ матрицы A (её формат NZ для удобного конвертирования в ZZ)
        // https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/opdevg/Ascendcopdevg/atlas_ascendc_10_0099.html#ZH-CN_TOPIC_0000002446676462__section184471251122117
        auto a1 = in_queue_A1.AllocTensor<float>(); 
        
        // выделили тензор для хранения БОЛЬШОЙ матрицы B (её формат NZ для удобного конвертирования в ZN)
        // https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/opdevg/Ascendcopdevg/atlas_ascendc_10_0099.html#ZH-CN_TOPIC_0000002446676462__section184471251122117
        auto b1 = in_queue_B1.AllocTensor<float>();

        // необходимо конвертировать А(B) в формат NZ (чтобы потом запихнуть в L1 cache и оттуда уже конвертировать в другой формат)
        CopyND2NZ(a1, global_matrix_a, n, n);
        CopyND2NZ(b1, global_matrix_b, n, n);

        in_queue_A1.EnQue(a1); // сохраняем в очередь тензора
        in_queue_B1.EnQue(b1); // сохраняем в очередь тензора
    }

    __aicore__ inline void SplitA()
    {
        auto a1 = in_queue_A1.DeQue<float>(); // копируем из очереди матрицу БОЛЬШУЮ матрицу A
        auto a2 = in_queue_A2.AllocTensor<float>(); // выделяем память под маленький блок матрицы A

        uint32_t offset = 0;
        
        for (uint32_t i = 0; i < blocks; ++i)
        {
            AscendC::LoadData2DParams load_data_params;

            load_data_params.repeatTimes = blocks;
            load_data_params.srcStride = blocks;
            load_data_params.ifTranspose = false;

            AscendC::LoadData(a2[offset], a1[offset], load_data_params);

            offset += 16 * 16;
        }

        in_queue_A1.FreeTensor(a1);
        in_queue_A2.EnQue(a2);


    }

public:

    __aicore__ inline MatmulCustom(uint32_t n)
    {
        this->n = n;
        this->blocks = 2;
    }

    __aicore__ inline void Init(AscendC::TPipe *p, GM_ADDR matrix_a, GM_ADDR matrix_b, GM_ADDR matrix_c)
    {
        pipe = p;

        uint32_t block_idx = AscendC::GetBlockIdx();

        global_matrix_a.SetGlobalBuffer((__gm__ float*)matrix_a);
        global_matrix_b.SetGlobalBuffer((__gm__ float*)matrix_b);
    
        pipe->InitBuffer(in_queue_A1, 1, n * n * sizeof(float));
        pipe->InitBuffer(in_queue_B1, 1, n * n * sizeof(float));

        pipe->InitBuffer(in_queue_A2, 1, n * n * sizeof(float));

        AscendC::printf("Block idx: %u \n", block_idx);
    }

    __aicore__ inline void Process()
    {
        CopyIn(); // копируем A и B из глобальной памяти в A1 и B1, а также конвертируем ND в NZ для последующего удобного преобразования
        SplitA(); // разделяем матрицу A на блоки
    }

};


extern "C" __global__ __aicore__ void matmul_custom(GM_ADDR matrix_a, GM_ADDR matrix_b, GM_ADDR matrix_c, uint32_t n)
{
    AscendC::TPipe pipe;
    MatmulCustom op(n);
    op.Init(&pipe, matrix_a, matrix_b, matrix_c);
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
extern void matmul_custom_do(uint32_t block_dim, void* stream, uint8_t* matrix_a, uint8_t* matrix_b, uint8_t* matrix_c)
{
    matmul_custom<<<block_dim, nullptr, stream>>>(matrix_a, matrix_b, matrix_c);
}
#endif