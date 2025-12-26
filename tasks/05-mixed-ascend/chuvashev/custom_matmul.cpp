#include "kernel_operator.h"

struct TileInfo {

    uint32_t n;
    uint32_t num_ai_cores;
    uint32_t sizeof_value;
    uint32_t plate_size;
    uint32_t single_core_A;
    uint32_t single_core_B;
    uint32_t single_core_N;
    uint32_t base;
    uint32_t rows;
    uint32_t cols;
    uint32_t block_size;

};

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
    AscendC::TQue<AscendC::TPosition::B2, 1> in_queue_B2;
    
    uint32_t blocks;

    uint32_t n;
    uint32_t num_ai_cores;
    uint32_t sizeof_value;
    uint32_t plate_size;
    uint32_t single_core_A;
    uint32_t single_core_B;
    uint32_t single_core_N;
    uint32_t base;
    uint32_t rows;
    uint32_t cols;
    uint32_t block_size;


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
        CopyND2NZ(a1, global_matrix_a, single_core_A, single_core_N);
        CopyND2NZ(b1, global_matrix_b, single_core_N, single_core_B);

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

    __aicore__ inline MatmulCustom(TileInfo *tile_ptr)
    {
        this->blocks = 2;
        
        this->n = tile_ptr->n;
        this->num_ai_cores = tile_ptr->num_ai_cores;
        this->sizeof_value = tile_ptr->sizeof_value;
        this->plate_size = tile_ptr->plate_size;
        this->single_core_A = tile_ptr->single_core_A;
        this->single_core_B = tile_ptr->single_core_B;
        this->single_core_N = tile_ptr->single_core_N;
        this->base = tile_ptr->base;
        this->rows = tile_ptr->rows;
        this->cols = tile_ptr->cols;
        this->block_size = tile_ptr->block_size;

    }

    __aicore__ inline void Init(AscendC::TPipe *p, GM_ADDR matrix_a, GM_ADDR matrix_b, GM_ADDR matrix_c)
    {
        pipe = p;

        uint32_t block_idx = AscendC::GetBlockIdx();

        uint32_t offset_a = (block_idx / cols) * n * single_core_A;
        uint32_t offset_b = (block_idx % cols) * single_core_B;
        uint32_t offset_c = offset_a + offset_b;

        global_matrix_a.SetGlobalBuffer((__gm__ float*)matrix_a + offset_a); // на основе нужного block_idx получаем необходимый адрес на строчку в A
        global_matrix_b.SetGlobalBuffer((__gm__ float*)matrix_b + offset_b); // на основе нужного block_idx получаем необходимый адрес на колонку в B
        global_matrix_c.SetGlobalBuffer((__gm__ float*)matrix_c + offset_c); // на основе нужного block_idx получаем необходимый адрес в C
    
        pipe->InitBuffer(in_queue_A1, 1, single_core_A * single_core_N * sizeof(float));
        pipe->InitBuffer(in_queue_B1, 1, single_core_B * single_core_N * sizeof(float));

        pipe->InitBuffer(in_queue_A2, 1, plate_size * plate_size * sizeof(float));
        pipe->InitBuffer(in_queue_B2, 1, plate_size * plate_size * sizeof(float));

        AscendC::printf("Block idx: %u\tOffset_A %u\tOffset_B %u\tOffset_C %u\n", block_idx, offset_a, offset_b, offset_c);
    }

    __aicore__ inline void Process()
    {
        CopyIn(); // копируем A и B из глобальной памяти в A1 и B1, а также конвертируем ND в NZ для последующего удобного преобразования
        // SplitA(); // разделяем матрицу A на блоки
    }

};


extern "C" __global__ __aicore__ void matmul_custom(GM_ADDR matrix_a, GM_ADDR matrix_b, GM_ADDR matrix_c, GM_ADDR tiling)
{

    TileInfo tile;

    tile.n = ((__gm__ TileInfo *)tiling)->n;

    tile.n = ((__gm__ TileInfo*)tiling)->n;
    tile.num_ai_cores = ((__gm__ TileInfo*)tiling)->num_ai_cores;
    tile.sizeof_value = ((__gm__ TileInfo*)tiling)->sizeof_value;
    tile.plate_size = ((__gm__ TileInfo*)tiling)->plate_size;
    tile.single_core_A = ((__gm__ TileInfo*)tiling)->single_core_A;
    tile.single_core_B = ((__gm__ TileInfo*)tiling)->single_core_B;
    tile.single_core_N = ((__gm__ TileInfo*)tiling)->single_core_N;
    tile.base = ((__gm__ TileInfo*)tiling)->base;
    tile.rows = ((__gm__ TileInfo*)tiling)->rows;
    tile.cols = ((__gm__ TileInfo*)tiling)->cols;
    tile.block_size = ((__gm__ TileInfo*)tiling)->block_size;

    AscendC::TPipe pipe;
    MatmulCustom op(&tile);
    op.Init(&pipe, matrix_a, matrix_b, matrix_c);
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
extern void matmul_custom_do(uint32_t block_dim, void* stream, uint8_t* matrix_a, uint8_t* matrix_b, uint8_t* matrix_c, uint8_t* tiling)
{
    matmul_custom<<<block_dim, nullptr, stream>>>(matrix_a, matrix_b, matrix_c, tiling);
}
#endif