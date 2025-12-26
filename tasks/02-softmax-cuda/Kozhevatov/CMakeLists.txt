cmake_minimum_required(VERSION 3.18)
project(softmax_cuda LANGUAGES CXX CUDA)

# Настройка версии 
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Настройка для производительности 
set(CMAKE_CUDA_ARCHITECTURES "native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")

# Настройка имен целевых файлов
set(target_suffix Kozhevatov)
set(target_name "softmax_cuda_${target_suffix}")

# Создание исполняемого файла
add_executable(${target_name} 
    main.cu
)

# Настройка свойств CUDA
set_target_properties(${target_name}
    PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_RUNTIME_LIBRARY "Static"
)

# Настройка свойств компиляции
target_compile_features(${target_name} PRIVATE cxx_std_17)

# Поиск и линковка CUDA
find_package(CUDAToolkit REQUIRED)
target_link_libraries(${target_name} 
    PRIVATE 
        CUDA::cudart
)

# Настройка для релизной сборки
if(CMAKE_BUILD_TYPE MATCHES Release)
    target_compile_options(${target_name} PRIVATE -O3)
endif()

# Добавляем целевые объекты для сборки
message(STATUS "Building target: ${target_name}")
message(STATUS "CUDA found: ${CUDAToolkit_FOUND}")
message(STATUS "CUDA version: ${CUDAToolkit_VERSION}")

# Функция для проверки архитектуры GPU 
function(check_gpu_architecture)
    execute_process(
        COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
        OUTPUT_VARIABLE COMPUTE_CAPABILITY
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    message(STATUS "GPU Compute Capability: ${COMPUTE_CAPABILITY}")
endfunction()

check_gpu_architecture()