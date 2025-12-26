cmake_minimum_required(VERSION 3.18)
project(matmul_softmax LANGUAGES CXX CUDA)

# Настройка версии C++/CUDA
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Настройка для Tensor Cores
set(CMAKE_CUDA_ARCHITECTURES "75")

# Настройка флагов компиляции
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math --expt-relaxed-constexpr")

# Настройка имен целевых файлов
set(target_suffix Kozhevatov)
set(target_name "matmul_softmax_${target_suffix}")

# Добавляем исполняемый файл
add_executable(${target_name} main.cu)

# Настройка свойств CUDA
set_target_properties(${target_name}
    PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Настройка свойств компиляции
target_compile_features(${target_name} PRIVATE cxx_std_17)

# Поиск и линковка OpenMP
find_package(OpenMP REQUIRED)
target_link_libraries(${target_name} PRIVATE OpenMP::OpenMP_CXX)

# Поиск и линковка CUDA
find_package(CUDAToolkit REQUIRED)
target_link_libraries(${target_name} PRIVATE CUDA::cudart)

# Включаем CUTLASS (header-only)
set(CUTLASS_ROOT "/opt/cutlass" CACHE PATH "Path to the CUTLASS installation")
if(EXISTS "${CUTLASS_ROOT}/include")
    target_include_directories(${target_name} PRIVATE "${CUTLASS_ROOT}/include")
    message(STATUS "Found CUTLASS at: ${CUTLASS_ROOT}")
    
    # Для CUTLASS нужны дополнительные флаги
    target_compile_options(${target_name} PRIVATE 
        $<$<COMPILE_LANGUAGE:CUDA>:
            --expt-relaxed-constexpr
            --expt-extended-lambda
            -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1
        >
    )
else()
    message(WARNING "CUTLASS not found at ${CUTLASS_ROOT}")
endif()