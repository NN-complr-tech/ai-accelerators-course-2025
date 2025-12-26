/**
 * @file main.cu
 * @brief CUDA реализация Softmax с использованием GPU (NVIDIA CUDA)
 * 
 * Программа вычисляет Softmax для каждой строки матрицы n×n на GPU.
 * Особенности реализации:
 * 1. Использование shared memory для эффективной редукции в пределах строки
 * 2. Двойная редукция (максимум и сумма) для численной стабильности
 * 3. Оптимизированное управление потоками CUDA (1 блок = 1 строка матрицы)
 * 4. Точное измерение времени выполнения через CUDA Events
 * 
 * Ограничения:
 * - Максимальный размер матрицы: 1024×1024 (ограничение CUDA: 1024 threads/block)
 * - Требуется GPU с поддержкой CUDA Compute Capability 3.5+
 * 
 * @param[in] argc Количество аргументов командной строки
 * @param[in] argv Аргументы командной строки
 * @return EXIT_SUCCESS при успешном выполнении, EXIT_FAILURE при ошибке
 * 
 * Пример использования:
 * @code{.sh}
 * ./softmax_cuda_Kozhevatov 256     # Тест с матрицей 256×256
 * ./softmax_cuda_Kozhevatov 1024    # Максимальный размер
 * @endcode
 */

// ===================== БИБЛИОТЕКИ И ИХ НАЗНАЧЕНИЕ =====================

#include <cuda_runtime.h>     // Основной API CUDA: cudaMalloc, cudaMemcpy, cudaFree
                              // И функции времени: cudaEventCreate, cudaEventRecord
#include <cfloat>             // Константы для чисел с плавающей точкой: FLT_MAX

#include <chrono>             // Измерение времени на CPU: high_resolution_clock
#include <cmath>              // Математические функции: std::exp, std::abs
#include <cstdlib>            // Базовые утилиты: EXIT_SUCCESS, EXIT_FAILURE
#include <iostream>           // Основной ввод-вывод: std::cout, std::cerr
#include <random>             // Генерация случайных чисел: std::mt19937
#include <vector>             // Динамический массив: std::vector<float>
#include <iomanip>            // Форматирование вывода: std::setprecision, std::fixed

// ===================== МАКРОС ДЛЯ ОБРАБОТКИ ОШИБОК CUDA =====================
/**
 * @brief Макрос для проверки ошибок CUDA API
 * 
 * Оборачивает вызов CUDA функции, проверяет возвращаемое значение и выводит
 * подробную информацию об ошибке при её возникновении.
 * 
 * @param call Вызов функции CUDA API
 */
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = (call); \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(error) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

namespace {

// ===================== ГЕНЕРАЦИЯ ТЕСТОВЫХ ДАННЫХ =====================
/**
 * @brief Генерация тестовой матрицы n×n со случайными значениями
 * 
 * Использует фиксированный seed (15) для воспроизводимости результатов.
 * Значения равномерно распределены в интервале [-1.0, 1.0].
 * 
 * @param n Размер матрицы (n×n)
 * @return Вектор размером n×n со случайными значениями
 */
std::vector<float> make_matrix(std::size_t n) {
    std::vector<float> matrix(n * n);
    std::mt19937 gen(15);  // Фиксированный seed для воспроизводимости
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (auto &x : matrix) {
        x = dist(gen);
    }
    return matrix;
}

// ===================== CPU ВЕРСИЯ (ПОСЛЕДОВАТЕЛЬНАЯ) =====================
/**
 * @brief Вычисление Softmax для одной строки (CPU, последовательная версия)
 * 
 * Алгоритм:
 * 1. Нахождение максимума в строке (для численной стабильности)
 * 2. Вычисление экспонент с смещением
 * 3. Вычисление суммы экспонент
 * 4. Нормализация (деление каждого элемента на сумму)
 * 
 * @param row Указатель на начало строки во входной матрице
 * @param result Указатель на место для результата
 * @param n Длина строки (количество столбцов)
 */
void softmax_row_sequential(const float* row, float* result, std::size_t n) {
    // Находим максимум для численной стабильности
    float max_val = row[0];
    for (std::size_t j = 1; j < n; ++j) {
        if (row[j] > max_val) {
            max_val = row[j];
        }
    }
    
    // Вычисляем экспоненты и их сумму
    float sum = 0.0f;
    for (std::size_t j = 0; j < n; ++j) {
        result[j] = std::exp(row[j] - max_val);
        sum += result[j];
    }
    
    // Нормализация
    float inv_sum = 1.0f / sum;
    for (std::size_t j = 0; j < n; ++j) {
        result[j] *= inv_sum;
    }
}

/**
 * @brief Вычисление Softmax для всей матрицы (CPU, последовательная версия)
 * 
 * Применяет softmax_row_sequential к каждой строке матрицы.
 * 
 * @param matrix Входная матрица (размера n×n)
 * @param n Размер матрицы
 * @return Результирующая матрица после применения Softmax к каждой строке
 */
std::vector<float> softmax_sequential(const std::vector<float>& matrix, std::size_t n) {
    std::vector<float> result(n * n);
    for (std::size_t i = 0; i < n; ++i) {
        softmax_row_sequential(&matrix[i * n], &result[i * n], n);
    }
    return result;
}

// ===================== CUDA ЯДРО =====================
/**
 * @brief CUDA ядро для вычисления Softmax
 * 
 * Каждый блок GPU обрабатывает одну строку матрицы.
 * Каждый поток в блоке обрабатывает один элемент строки.
 * 
 * Алгоритм в shared memory:
 * 1. Загрузка данных из глобальной памяти в shared memory
 * 2. Редукция для нахождения максимума в строке
 * 3. Вычисление экспонент с учётом максимума
 * 4. Редукция для вычисления суммы экспонент
 * 5. Нормализация и запись результата
 * 
 * @param output Указатель на выходную матрицу в глобальной памяти GPU
 * @param input Указатель на входную матрицу в глобальной памяти GPU
 * @param n Размер матрицы (n×n)
 * 
 * @note Используется dynamic shared memory: extern __shared__ float shared_mem[]
 * @note Размер shared memory должен быть не менее n * sizeof(float) байт
 */
__global__ void softmax_kernel(float* output, const float* input, std::size_t n) {
    // Индекс строки (блок)
    int row_idx = blockIdx.x;
    // Индекс столбца (поток в блоке)
    int tid = threadIdx.x;
    
    // Используем shared memory для редукции
    extern __shared__ float shared_mem[];
    
    // 1. Загружаем значение из глобальной памяти
    float val = (tid < n) ? input[row_idx * n + tid] : -FLT_MAX;
    
    // 2. Редукция для нахождения максимума в строке
    shared_mem[tid] = val;
    __syncthreads();
    
    // Редукция в shared memory (дерево)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float other = shared_mem[tid + s];
            if (other > shared_mem[tid]) {
                shared_mem[tid] = other;
            }
        }
        __syncthreads();
    }
    
    float row_max = shared_mem[0];
    __syncthreads();
    
    // 3. Вычисляем экспоненты с учетом максимума для стабильности
    float exp_val = 0.0f;
    if (tid < n) {
        exp_val = expf(val - row_max);
    }
    
    // 4. Редукция для суммы экспонент
    shared_mem[tid] = exp_val;
    __syncthreads();
    
    // Редукция суммы в shared memory (дерево)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_mem[tid] += shared_mem[tid + s];
        }
        __syncthreads();
    }
    
    float row_sum = shared_mem[0];
    
    // 5. Нормализация и запись результата
    if (tid < n) {
        output[row_idx * n + tid] = exp_val / row_sum;
    }
}

// ===================== ОБВЕРТКА ДЛЯ CUDA =====================
/**
 * @brief Обертка для запуска CUDA ядра и управления памятью GPU
 * 
 * Выполняет:
 * 1. Выделение памяти на GPU
 * 2. Копирование данных CPU→GPU
 * 3. Конфигурацию и запуск ядра
 * 4. Измерение времени выполнения через CUDA Events
 * 5. Копирование результатов GPU→CPU
 * 6. Освобождение ресурсов
 * 
 * @param matrix Входная матрица (на CPU)
 * @param n Размер матрицы
 * @param elapsed_time[out] Время выполнения ядра (в секундах)
 * @param warmup Флаг прогревочного запуска (не измеряется)
 * @return Результирующая матрица после Softmax
 */
std::vector<float> softmax_cuda(const std::vector<float>& matrix, std::size_t n, 
                                double& elapsed_time, bool warmup = false) {
    std::size_t size = n * n * sizeof(float);
    
    // Выделяем память на устройстве
    float* d_input = nullptr;
    float* d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, size));
    CUDA_CHECK(cudaMalloc(&d_output, size));
    
    // Копируем данные на устройство
    CUDA_CHECK(cudaMemcpy(d_input, matrix.data(), size, cudaMemcpyHostToDevice));
    
    // Конфигурация запуска
    dim3 block_size(n, 1, 1);  // n потоков в блоке
    dim3 grid_size(n, 1, 1);   // n блоков (по одному на строку)
    
    // Размер shared memory
    size_t shared_mem_size = n * sizeof(float);
    
    // Создаем события для замера времени
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    if (warmup) {
        // Прогревочный запуск (не замеряем)
        softmax_kernel<<<grid_size, block_size, shared_mem_size>>>(d_output, d_input, n);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    // Основной запуск с замером времени
    CUDA_CHECK(cudaEventRecord(start));
    softmax_kernel<<<grid_size, block_size, shared_mem_size>>>(d_output, d_input, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    // Вычисляем время
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    elapsed_time = milliseconds / 1000.0;
    
    // Копируем результат обратно
    std::vector<float> result(n * n);
    CUDA_CHECK(cudaMemcpy(result.data(), d_output, size, cudaMemcpyDeviceToHost));
    
    // Освобождаем ресурсы
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return result;
}

// ===================== ВАЛИДАЦИЯ РЕЗУЛЬТАТОВ =====================
/**
 * @brief Вычисление максимальной абсолютной разницы между двумя матрицами
 * 
 * Используется для проверки корректности CUDA реализации
 * путём сравнения с CPU эталоном.
 * 
 * @param a Первая матрица
 * @param b Вторая матрица
 * @return Максимальная абсолютная разница между соответствующими элементами
 */
float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        std::cerr << "Error: vectors size mismatch" << std::endl;
        exit(EXIT_FAILURE);
    }
    
    float max_diff = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) {
        float diff = std::abs(a[i] - b[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    return max_diff;
}

/**
 * @brief Проверка, что суммы элементов каждой строки равны 1.0 (с допуском)
 * 
 * Softmax гарантирует, что сумма элементов каждой строки = 1.0
 * Эта проверка валидирует численную корректность реализации.
 * 
 * @param matrix Проверяемая матрица
 * @param n Размер матрицы
 * @param tolerance Допустимое отклонение от 1.0
 * @return true если все строки имеют сумму 1.0±tolerance, иначе false
 */
bool validate_row_sums(const std::vector<float>& matrix, std::size_t n, float tolerance = 1e-4f) {
    bool all_valid = true;
    for (std::size_t i = 0; i < n; ++i) {
        float sum = 0.0f;
        for (std::size_t j = 0; j < n; ++j) {
            sum += matrix[i * n + j];
        }
        if (std::abs(sum - 1.0f) > tolerance) {
            std::cout << "WARNING: Row " << i << " sum = " << std::fixed << std::setprecision(6) 
                      << sum << " (expected ~1.0)" << std::endl;
            all_valid = false;
        }
    }
    return all_valid;
}

} // namespace

// ===================== ОСНОВНАЯ ФУНКЦИЯ =====================
/**
 * @brief Точка входа в программу
 * 
 * Выполняет:
 * 1. Парсинг аргументов командной строки
 * 2. Генерацию тестовой матрицы
 * 3. Запуск CPU и GPU версий
 * 4. Сравнение результатов и вывод производительности
 * 
 * @param argc Количество аргументов
 * @param argv Аргументы (ожидается: имя_программы размер_матрицы)
 * @return Код завершения: EXIT_SUCCESS или EXIT_FAILURE
 */
int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <matrix_size_n>" << std::endl;
        std::cerr << "  n must be <= 1024 (maximum threads per block)" << std::endl;
        return EXIT_FAILURE;
    }

    std::size_t n = std::stoul(argv[1]);
    if (n == 0) {
        std::cerr << "Error: matrix size must be positive" << std::endl;
        return EXIT_FAILURE;
    }
    
    if (n > 1024) {
        std::cerr << "Error: this implementation supports n <= 1024" << std::endl;
        std::cerr << "  (maximum threads per CUDA block is 1024)" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Matrix size: " << n << "x" << n << std::endl;
    std::cout << "Generating random matrix..." << std::endl;
    
    // Генерируем матрицу
    auto matrix = make_matrix(n);
    
    // ===== SEQUENTIAL VERSION =====
    std::cout << "\nRunning sequential version..." << std::endl;
    auto start_seq = std::chrono::high_resolution_clock::now();
    auto result_seq = softmax_sequential(matrix, n);
    auto stop_seq = std::chrono::high_resolution_clock::now();
    double time_seq = std::chrono::duration<double>(stop_seq - start_seq).count();
    
    // ===== CUDA VERSION =====
    std::cout << "Running CUDA version..." << std::endl;
    
    // Прогревочный запуск (исключаем из измерений)
    std::cout << "  Warming up GPU..." << std::endl;
    double warmup_time;
    softmax_cuda(matrix, n, warmup_time, true);
    
    // Основной запуск с измерением
    double time_cuda;
    auto result_cuda = softmax_cuda(matrix, n, time_cuda, false);
    
    // ===== VALIDATION =====
    std::cout << "\nValidating results..." << std::endl;
    float diff = max_abs_diff(result_seq, result_cuda);
    
    // Проверяем суммы строк для обоих результатов
    bool cuda_sums_valid = validate_row_sums(result_cuda, n);
    bool cpu_sums_valid = validate_row_sums(result_seq, n);
    
    // ===== OUTPUT RESULTS =====
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\n=========================================" << std::endl;
    std::cout << "Sequential: " << time_seq << " sec" << std::endl;
    std::cout << "SIMT: " << time_cuda << " sec (diff: ";
    
    // Форматируем разницу
    if (diff < 1e-10) {
        std::cout << "< 1e-10";
    } else {
        std::cout << std::scientific << std::setprecision(2) << diff;
    }
    std::cout << ")" << std::endl;
    
    // Ускорение
    if (time_seq > 0 && time_cuda > 0) {
        double speedup = time_seq / time_cuda;
        std::cout << "Speedup: " << std::fixed << std::setprecision(2) 
                  << speedup << "x" << std::endl;
    }
    
    std::cout << "=========================================" << std::endl;
    
    // ===== Дополнительная информация =====
    std::cout << "\nAdditional checks:" << std::endl;
    std::cout << "  - CUDA row sums valid: " << (cuda_sums_valid ? "YES" : "NO") << std::endl;
    std::cout << "  - CPU row sums valid: " << (cpu_sums_valid ? "YES" : "NO") << std::endl;
    
    if (cuda_sums_valid && cpu_sums_valid && diff < 1e-5f) {
        std::cout << "\n SUCCESS: All validations passed!" << std::endl;
        std::cout << "   Results are numerically correct." << std::endl;
    } else {
        std::cout << "\n WARNING: Some validations failed!" << std::endl;
        if (diff >= 1e-5f) {
            std::cout << "   CPU and GPU results differ significantly." << std::endl;
        }
        if (!cuda_sums_valid) {
            std::cout << "   CUDA row sums are not equal to 1.0" << std::endl;
        }
        if (!cpu_sums_valid) {
            std::cout << "   CPU row sums are not equal to 1.0" << std::endl;
        }
    }
    
    return EXIT_SUCCESS;
}