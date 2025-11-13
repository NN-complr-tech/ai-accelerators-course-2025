enable_language(CUDA CXX)

set(target_suffix chuvashev)
set(target_name "softmax_cuda_${target_suffix}")

add_executable(${target_name} 
    src/kernel.cu 
    src/main.cpp
    src/utils.cpp
    include/utils.h
)

source_group("Source Files" FILES 
    src/kernel.cu 
    src/main.cpp
    src/utils.cpp
)
source_group("Header Files" FILES 
    include/utils.h
)

target_include_directories(${target_name} PRIVATE include/)

find_package(OpenMP REQUIRED)
target_link_libraries(${target_name} PRIVATE OpenMP::OpenMP_CXX)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(${target_name} PRIVATE CUDA::cudart)

target_compile_features(${target_name} PRIVATE cxx_std_17)

set_target_properties(${target_name}
    PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
)