#include <iostream>
#include "data_utils.h"

#ifndef ASCENDC_CPU_DEBUG

#include "acl/acl.h"
extern void exp_custom_do(uint32_t block_dim, void* stream, uint8_t* x, uint8_t* y, uint32_t n);

#else

#include "tikicpulib.h"
extern "C" __global__ __aicore__ void exp_custom(GM_ADDR x, GM_ADDR y, uint32_t n);

#endif

int main()
{
    uint32_t n = 512;
    uint32_t min_rows_per_core = 128;
    uint32_t block_dim = (std::min)(8, int((n + min_rows_per_core - 1) / min_rows_per_core));
    std::size_t input_count_of_bytes = n * n * sizeof(float);
    std::size_t output_count_of_bytes = n * n * sizeof(float);

    #ifdef ASCENDC_CPU_DEBUG

    std::cout << "SIZE IN MB: " << (float)input_count_of_bytes / (1024.0 * 1024.0) << std::endl; 

    uint8_t *x = (uint8_t*)AscendC::GmAlloc(input_count_of_bytes);
    uint8_t *y = (uint8_t*)AscendC::GmAlloc(input_count_of_bytes);

    ReadFile("./input/input_x.bin", input_count_of_bytes, x, input_count_of_bytes);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(exp_custom, block_dim, x, y, n);

    WriteFile("./output/output_y.bin", y, output_count_of_bytes);

    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)y);

    #else

    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));

    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    uint8_t* x_host, *y_host;
    uint8_t* x_device, *y_device;

    CHECK_ACL(aclrtMallocHost((void**)(&x_host), input_count_of_bytes));
    CHECK_ACL(aclrtMallocHost((void**)(&y_host), input_count_of_bytes));

    ReadFile("./input/input_x.bin", input_count_of_bytes, x_host, input_count_of_bytes);
    
    CHECK_ACL(aclrtMalloc((void**)(&x_device), input_count_of_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void**)(&y_device), input_count_of_bytes, ACL_MEM_MALLOC_HUGE_FIRST));

    CHECK_ACL(aclrtMemcpy(x_device, input_count_of_bytes, x_host, input_count_of_bytes, ACL_MEMCPY_HOST_TO_DEVICE));

    exp_custom_do(block_dim, stream, x_device, y_device, n);

    CHECK_ACL(aclrtSynchronizeStream(stream));

    CHECK_ACL(aclrtMemcpy(y_host, output_count_of_bytes, y_device, output_count_of_bytes, ACL_MEMCPY_DEVICE_TO_HOST));

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