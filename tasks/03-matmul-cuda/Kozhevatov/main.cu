/**
 * @file cuda_softmax_benchmark.cpp
 * @brief Код внешне большой из - за дебага --debug
 * 
 * Программа сравнивает три подхода к вычислению Softmax(A×B) для матриц размером n×n:
 * 1. OpenMP (CPU) - эталонная реализация на CPU с использованием OpenMP
 * 2. WMMA (Tensor Cores) - реализация с использованием Tensor Cores через WMMA API
 * 3. CUTLASS - высокооптимизированная реализация с использованием библиотеки CUTLASS
 * 
 * @param[in] argc Количество аргументов командной строки
 * @param[in] argv Аргументы командной строки
 * @return EXIT_SUCCESS при успешном выполнении, EXIT_FAILURE при ошибке
 * 
 * Пример использования:
 * @code{.sh}
 * ./cuda_softmax_benchmark 1024           # Тест с матрицей 1024x1024
 * ./cuda_softmax_benchmark 512 --debug    # Тест с отладочной информацией
 * @endcode
 */

#ifndef CUDA_NO_BFLOAT16
#define CUDA_NO_BFLOAT16 1  ///< Отключает поддержку bfloat16 для совместимости
#endif

#include <cuda.h>            ///< Основной заголовочный файл CUDA (драйверный API)
#include <cuda_fp16.h>       ///< Поддержка половинной точности (FP16) в CUDA
#include <cuda_runtime.h>    ///< Runtime API CUDA для управления памятью и выполнения ядер
#include <mma.h>             ///< Warp Matrix Multiply-Accumulate (WMMA) API для Tensor Cores

#include <algorithm>         ///< Алгоритмы STL: std::min, std::max, std::generate
#include <chrono>            ///< Измерение времени: high_resolution_clock, duration
#include <cmath>             ///< Математические функции: std::exp, std::abs, std::isnan
#include <cfloat>            ///< Константы для чисел с плавающей точкой: FLT_MAX
#include <cstdlib>           ///< Стандартные утилиты: EXIT_SUCCESS, EXIT_FAILURE
#include <exception>         ///< Обработка исключений: std::exception, std::runtime_error
#include <functional>        ///< Функциональные объекты: std::function для коллбэков
#include <iomanip>           ///< Форматирование вывода: std::setprecision, std::fixed
#include <iostream>          ///< Потоки ввода-вывода: std::cout, std::cerr
#include <random>            ///< Генерация случайных чисел: std::mt19937, std::normal_distribution
#include <sstream>           ///< Строковые потоки: std::ostringstream для форматирования
#include <stdexcept>         ///< Стандартные исключения: std::invalid_argument
#include <string>            ///< Строки std::string
#include <vector>            ///< Динамический массив std::vector для хранения матриц

#include "cutlass/gemm/device/gemm.h"  ///< Библиотека CUTLASS для высокооптимизированного GEMM



/**
 * @def CHECK_CUDA_ERROR
 * @brief Макрос для проверки ошибок CUDA
 * 
 * Оборачивает вызов CUDA функций для автоматической проверки кода возврата.
 * При обнаружении ошибки выводит подробную информацию и завершает программу.
 * 
 * @param callable Выражение, возвращающее cudaError_t
 */
#define CHECK_CUDA_ERROR(callable)                                        \
  {                                                                       \
    auto codeError = callable;                                            \
    if (codeError != cudaSuccess) {                                       \
      std::cerr << "\033[1;31merror\033[0m: ";                            \
      std::cerr << cudaGetErrorString(codeError) << '\n';                 \
      std::cerr << "code error: " << static_cast<int>(codeError) << '\n'; \
      std::cerr << "loc: " << __FILE__ << '(' << __LINE__ << ")\n";       \
      std::exit(codeError);                                               \
    }                                                                     \
  }

namespace {
/**
 * @var DEBUG_MODE
 * @brief Глобальный флаг для управления отладочным выводом
 * 
 * При установке в true активирует вывод промежуточных результатов,
 * проверки корректности и дополнительной информации для отладки.
 */
bool DEBUG_MODE = false;

// ===================== ДЕБАГ-ФУНКЦИИ =====================

/**
 * @brief Выводит часть матрицы для отладки
 * 
 * Позволяет визуально проверить содержимое матрицы на этапе отладки.
 * Выводит заданное количество первых строк и столбцов.
 * 
 * @param[in] matrix Вектор, содержащий матрицу в row-major порядке
 * @param[in] n Размер матрицы (n×n)
 * @param[in] name Имя матрицы для вывода в заголовке
 * @param[in] rows_to_print Количество строк для вывода (по умолчанию 3)
 * @param[in] cols_to_print Количество столбцов для вывода (по умолчанию 5)
 */
void debug_print_matrix(const std::vector<float>& matrix, std::size_t n, 
                       const std::string& name, int rows_to_print = 3, int cols_to_print = 5) {
  if (!DEBUG_MODE) return;
  
  std::cout << "\n=== " << name << " (first " << rows_to_print << " rows) ===" << std::endl;
  std::cout << std::fixed << std::setprecision(6);
  
  for (std::size_t i = 0; i < std::min(n, (std::size_t)rows_to_print); ++i) {
    std::cout << "Row " << i << ": ";
    for (std::size_t j = 0; j < std::min(n, (std::size_t)cols_to_print); ++j) {
      std::cout << std::setw(12) << matrix[i * n + j] << " ";
    }
    if (cols_to_print < n) std::cout << "...";
    std::cout << std::endl;
  }
}

/**
 * @brief Выводит часть матрицы в формате половинной точности (FP16)
 * 
 * Аналогично debug_print_matrix, но для матриц в формате __half.
 * 
 * @param[in] matrix Вектор матрицы в формате __half
 * @param[in] n Размер матрицы (n×n)
 * @param[in] name Имя матрицы для вывода
 * @param[in] offset Смещение в векторе для начала вывода
 */
void debug_print_matrix_half(const std::vector<__half>& matrix, std::size_t n,
                            const std::string& name, int offset = 0) {
  if (!DEBUG_MODE) return;
  
  std::cout << "\n=== " << name << " (first 3x3) ===" << std::endl;
  std::cout << std::fixed << std::setprecision(4);
  
  for (std::size_t i = 0; i < std::min(n, (std::size_t)3); ++i) {
    std::cout << "  ";
    for (std::size_t j = 0; j < std::min(n, (std::size_t)3); ++j) {
      std::cout << std::setw(8) << __half2float(matrix[offset + i * n + j]) << " ";
    }
    std::cout << std::endl;
  }
}

/**
 * @brief Проверяет суммы строк матрицы после softmax
 * 
 * После применения softmax сумма элементов каждой строки должна быть равна 1.
 * Эта функция проверяет это условие для первых нескольких строк.
 * 
 * @param[in] matrix Проверяемая матрица
 * @param[in] n Размер матрицы (n×n)
 * @param[in] name Имя матрицы для вывода результатов
 */
void debug_check_row_sums(const std::vector<float>& matrix, std::size_t n,
                         const std::string& name) {
  if (!DEBUG_MODE) return;
  
  std::cout << "\n=== Row sums check for " << name << " ===" << std::endl;
  std::cout << std::fixed << std::setprecision(6);
  
  int errors = 0;
  for (std::size_t i = 0; i < std::min(n, (std::size_t)5); ++i) {
    float sum = 0.0f;
    for (std::size_t j = 0; j < n; ++j) {
      sum += matrix[i * n + j];
    }
    
    bool ok = std::abs(sum - 1.0f) < 1e-4f;
    std::cout << "  Row " << i << ": sum = " << sum 
              << (ok ? " [OK]" : " [ERROR!]") << std::endl;
    
    if (!ok) errors++;
  }
  
  if (errors > 0) {
    std::cout << "  WARNING: " << errors << " rows have incorrect sums!" << std::endl;
  }
}

/**
 * @brief Сравнивает две матрицы поэлементно
 * 
 * Вычисляет разницу между двумя матрицами и выводит статистику расхождений.
 * Полезно для проверки корректности GPU реализаций относительно CPU эталона.
 * 
 * @param[in] a Первая матрица (обычно эталонная CPU реализация)
 * @param[in] b Вторая матрица (GPU реализация для проверки)
 * @param[in] n Размер матриц (n×n)
 * @param[in] name_a Имя первой матрицы
 * @param[in] name_b Имя второй матрицы
 */
void debug_compare_matrices(const std::vector<float>& a, const std::vector<float>& b,
                           std::size_t n, const std::string& name_a, const std::string& name_b) {
  if (!DEBUG_MODE) return;
  
  std::cout << "\n=== Comparison " << name_a << " vs " << name_b << " ===" << std::endl;
  
  int mismatches = 0;
  float max_diff = 0.0f;
  std::size_t max_diff_idx = 0;
  
  for (std::size_t i = 0; i < std::min((std::size_t)100, n * n); ++i) {
    float diff = std::abs(a[i] - b[i]);
    if (diff > max_diff) {
      max_diff = diff;
      max_diff_idx = i;
    }
    if (diff > 1e-4f) {
      mismatches++;
      if (mismatches <= 5) {
        std::cout << "  Mismatch [" << i/n << "," << i%n << "]: "
                  << name_a << "=" << a[i] << ", " << name_b << "=" << b[i]
                  << ", diff=" << diff << std::endl;
      }
    }
  }
  
  std::cout << "  Max difference: " << max_diff 
            << " at index [" << max_diff_idx/n << "," << max_diff_idx%n << "]" << std::endl;
  std::cout << "  Elements with diff > 1e-4: " << mismatches << " of " << n*n << std::endl;
}

// ===================== ГЕНЕРАЦИЯ ДАННЫХ =====================

/**
 * @brief Генерирует две случайные матрицы для тестирования
 * 
 * Создает две матрицы A и B размером n×n с элементами из нормального распределения.
 * Матрицы хранятся в формате половинной точности (__half) для совместимости с Tensor Cores.
 * 
 * @param[in] n Размер матриц (n×n)
 * @return Вектор, содержащий матрицы A и B в формате row-major: [A (n×n элементов), B (n×n элементов)]
 * 
 * @note Используется фиксированный seed (42) для воспроизводимости результатов
 */
std::vector<__half> make_input_matrix(std::size_t n) {
  static std::mt19937 gen_eng{42};
  std::normal_distribution<float> dist{0.0f, 1.0f};  // mean=0, stddev=1
  auto gen = [&]() { return static_cast<__half>(dist(gen_eng)); };

  std::vector<__half> matrix(n * n * 2);
  std::generate(matrix.begin(), matrix.end(), gen);

  // Debug: show input data
  debug_print_matrix_half(matrix, n, "Matrix A", 0);
  debug_print_matrix_half(matrix, n, "Matrix B", n * n);

  return matrix;
}

// ===================== OpenMP РЕАЛИЗАЦИЯ =====================

/**
 * @brief Эталонная реализация на CPU с использованием OpenMP
 * 
 * Выполняет матричное умножение A×B с последующим softmax для каждой строки.
 * Используется как эталон для проверки корректности GPU реализаций.
 * 
 * @param[in] matrix Входной вектор, содержащий матрицы A и B
 * @param[in] n Размер матриц (n×n)
 * @return Результирующая матрица после применения softmax к A×B
 * 
 * Алгоритм:
 * 1. Матричное умножение A×B (тройной вложенный цикл)
 * 2. Для каждой строки результата:
 *    - Нахождение максимума для численной стабильности
 *    - Вычисление экспонент и их суммы
 *    - Нормализация (деление каждого элемента на сумму)
 */
std::vector<float> run_openmp_reference(const std::vector<__half>& matrix,
                                        std::size_t n) {
  const size_t size = n * n;
  std::vector<float> result(size, 0.0f);
  const __half* a = matrix.data();
  const __half* b = matrix.data() + size;

  #pragma omp parallel for
  for (size_t i = 0; i < n; ++i) {
    float* row = &result[i * n];
    
    // Matrix multiplication
    for (size_t k = 0; k < n; ++k) {
      float a_ik = __half2float(a[i * n + k]);
      for (size_t j = 0; j < n; ++j) {
        row[j] += a_ik * __half2float(b[k * n + j]);
      }
    }
    
    // Debug: show intermediate values for first 2 rows
    if (DEBUG_MODE && i < 2) {
      std::cout << "\nOpenMP row " << i << " after multiplication (first 5): ";
      for (int j = 0; j < std::min(5, (int)n); ++j) {
        std::cout << row[j] << " ";
      }
      std::cout << std::endl;
    }
    
    // Softmax
    // 1. Find maximum for numerical stability
    float max_val = row[0];
    for (size_t j = 1; j < n; ++j) {
      if (row[j] > max_val) max_val = row[j];
    }
    
    // 2. Compute exponentials and sum
    float sum = 0.0f;
    for (size_t j = 0; j < n; ++j) {
      row[j] = std::exp(row[j] - max_val);
      sum += row[j];
    }
    
    // 3. Normalization
    float inv_sum = 1.0f / sum;
    for (size_t j = 0; j < n; ++j) {
      row[j] *= inv_sum;
    }
  }

  // Debug: show results
  debug_print_matrix(result, n, "OpenMP result");
  debug_check_row_sums(result, n, "OpenMP");
  
  return result;
}

// ===================== CUDA ЯДРО SOFTMAX =====================

/**
 * @brief CUDA ядро для вычисления softmax
 * 
 * Вычисляет softmax для каждой строки матрицы независимо.
 * Использует shared memory для эффективной редукции внутри блока.
 * 
 * @param[in] input Входная матрица (результат умножения A×B)
 * @param[out] output Выходная матрица после softmax
 * @param[in] n Размер матрицы (n×n)
 * 
 * Алгоритм для каждой строки:
 * 1. Нахождение максимума в строке (редукция через shared memory)
 * 2. Вычисление экспонент и их суммы (редукция через shared memory)
 * 3. Нормализация: деление каждой экспоненты на сумму
 * 
 * @note Каждый блок CUDA обрабатывает одну строку матрицы
 * @note Для численной стабильности используется формула: exp(x - max(x)) / sum(exp(x - max(x)))
 */
__global__ void softmax_kernel(const float* input, float* output, size_t n) {
  unsigned int row_idx = blockIdx.x;
  unsigned int tid = threadIdx.x;
  unsigned int block_size = blockDim.x;

  extern __shared__ float sdata[];
  const float* row = &input[row_idx * n];
  float* out = &output[row_idx * n];

  // 1. Find maximum
  float max_val = -FLT_MAX;
  for (unsigned int j = tid; j < n; j += block_size) {
    max_val = fmaxf(max_val, row[j]);
  }
  sdata[tid] = max_val;
  __syncthreads();

  // Reduction for maximum
  for (unsigned int s = block_size >> 1; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
    }
    __syncthreads();
  }
  float row_max = sdata[0];
  __syncthreads();

  // 2. Compute exp and sum
  float tid_sum = 0.0f;
  for (unsigned int j = tid; j < n; j += block_size) {
    tid_sum += __expf(row[j] - row_max);
  }
  sdata[tid] = tid_sum;
  __syncthreads();

  // Reduction for sum
  for (unsigned int s = block_size >> 1; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }
  float row_sum = sdata[0];
  __syncthreads();

  // 3. Normalization
  for (unsigned int j = tid; j < n; j += block_size) {
    out[j] = __expf(row[j] - row_max) / row_sum;
  }
}

// ===================== WMMA РЕАЛИЗАЦИЯ =====================

using namespace nvcuda;

/**
 * @var WMMA_SIZE
 * @brief Размер матричного блока для Tensor Cores (16×16)
 * 
 * Tensor Cores работают с блоками фиксированного размера 16×16×16.
 */
constexpr std::size_t WMMA_SIZE = 16;

/**
 * @var WARP_SIZE
 * @brief Размер warp'а в CUDA (32 потока)
 */
constexpr std::size_t WARP_SIZE = 32;

/**
 * @brief CUDA ядро для матричного умножения с использованием WMMA (Tensor Cores)
 * 
 * Реализует матричное умножение A×B с использованием Tensor Cores через WMMA API.
 * Каждый warp вычисляет один блок 16×16 результирующей матрицы.
 * 
 * @param[in] a Указатель на матрицу A в device memory
 * @param[in] b Указатель на матрицу B в device memory
 * @param[out] c Указатель на результирующую матрицу C в device memory
 * @param[in] n Размер матриц (n×n)
 * 
 * @note Использует row-major layout для всех матриц
 * @note Предполагает, что матрицы выровнены по границам 16 элементов
 */
__global__ void GEMMv4(const __half* a, const __half* b, float* c, std::size_t n) {
  wmma::fragment<wmma::matrix_a, WMMA_SIZE, WMMA_SIZE, WMMA_SIZE, half,
                 wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_SIZE, WMMA_SIZE, WMMA_SIZE, half,
                 wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, WMMA_SIZE, WMMA_SIZE, WMMA_SIZE, float> acc_frag;

  wmma::fill_fragment(acc_frag, 0.0f);
  int row = (blockIdx.y * blockDim.y + threadIdx.y) * WMMA_SIZE;
  int col = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE * WMMA_SIZE;

  if (row >= n || col >= n) return;

  for (int k = 0; k < n; k += WMMA_SIZE) {
    wmma::load_matrix_sync(a_frag, &a[row * n + k], n);
    wmma::load_matrix_sync(b_frag, &b[k * n + col], n);
    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
  }

  wmma::store_matrix_sync(&c[row * n + col], acc_frag, n, wmma::mem_row_major);
}

/**
 * @brief Реализация матричного умножения с softmax через WMMA (Tensor Cores)
 * 
 * Выполняет матричное умножение A×B с использованием Tensor Cores,
 * затем применяет softmax к результату.
 * 
 * @param[in] matrix Входной вектор, содержащий матрицы A и B
 * @param[in] n Размер матриц (n×n)
 * @return Результирующая матрица после применения softmax к A×B
 * 
 * Алгоритм:
 * 1. Выделение памяти на GPU и копирование данных с CPU
 * 2. Матричное умножение через WMMA (ядро GEMMv4)
 * 3. Применение softmax через softmax_kernel
 * 4. Копирование результата обратно на CPU
 * 
 * @note Для маленьких матриц (n ≤ 16) используется CPU реализация
 */
std::vector<float> run_wmma(const std::vector<__half>& matrix, std::size_t n) {
  if (n <= 16) return run_openmp_reference(matrix, n);

  const size_t size = n * n;
  std::vector<float> res(size);
  
  // Debug: buffer for intermediate data (multiplication result)
  std::vector<float> gemm_result(size);

  half *a_dev = nullptr, *b_dev = nullptr;
  float* c_dev = nullptr;
  float* temp_dev = nullptr;

  CHECK_CUDA_ERROR(cudaMalloc(&a_dev, size * sizeof(half)));
  CHECK_CUDA_ERROR(cudaMalloc(&b_dev, size * sizeof(half)));
  CHECK_CUDA_ERROR(cudaMalloc(&c_dev, size * sizeof(float)));
  CHECK_CUDA_ERROR(cudaMalloc(&temp_dev, size * sizeof(float)));

  // Initialize with zeros (important!)
  CHECK_CUDA_ERROR(cudaMemset(temp_dev, 0, size * sizeof(float)));
  CHECK_CUDA_ERROR(cudaMemset(c_dev, 0, size * sizeof(float)));

  CHECK_CUDA_ERROR(cudaMemcpy(a_dev, matrix.data(), size * sizeof(half),
                              cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(b_dev, matrix.data() + size, size * sizeof(half),
                              cudaMemcpyHostToDevice));

  // Configuration for WMMA
  constexpr int THREADS_PER_BLOCK = 256;
  constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / WARP_SIZE;
  
  dim3 block_size(THREADS_PER_BLOCK);
  const int tiles_per_dim = static_cast<int>((n + WMMA_SIZE - 1) / WMMA_SIZE);
  dim3 grid_size((tiles_per_dim + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK,
                 tiles_per_dim);

  if (DEBUG_MODE) {
    std::cout << "\n=== WMMA configuration ===" << std::endl;
    std::cout << "  Matrix size: " << n << "x" << n << std::endl;
    std::cout << "  Tiles per dim: " << tiles_per_dim << std::endl;
    std::cout << "  Block size: " << block_size.x << std::endl;
    std::cout << "  Grid size: (" << grid_size.x << ", " << grid_size.y << ")" << std::endl;
  }

  // 1. Matrix multiplication
  GEMMv4<<<grid_size, block_size>>>(a_dev, b_dev, temp_dev, n);
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());
  
  // Debug: copy intermediate multiplication result
  if (DEBUG_MODE) {
    CHECK_CUDA_ERROR(cudaMemcpy(gemm_result.data(), temp_dev, size * sizeof(float),
                                cudaMemcpyDeviceToHost));
    debug_print_matrix(gemm_result, n, "WMMA after multiplication");
  }

  // 2. Softmax
  const int softmax_threads = 256;
  dim3 softmax_block(softmax_threads);
  dim3 softmax_grid(n);
  size_t softmax_shared = softmax_threads * sizeof(float);
  softmax_kernel<<<softmax_grid, softmax_block, softmax_shared>>>(temp_dev, c_dev, n);
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  // Copy final result
  CHECK_CUDA_ERROR(cudaMemcpy(res.data(), c_dev, size * sizeof(float),
                              cudaMemcpyDeviceToHost));

  // Debug: show results
  if (DEBUG_MODE) {
    debug_print_matrix(res, n, "WMMA final result");
    debug_check_row_sums(res, n, "WMMA");
    
    // Check for NaN/Inf in results
    int nan_count = 0, inf_count = 0;
    for (const auto& val : res) {
      if (std::isnan(val)) nan_count++;
      if (std::isinf(val)) inf_count++;
    }
    if (nan_count > 0 || inf_count > 0) {
      std::cout << "  WARNING: WMMA contains " << nan_count << " NaN and " 
                << inf_count << " Inf values!" << std::endl;
    }
  }

  CHECK_CUDA_ERROR(cudaFree(a_dev));
  CHECK_CUDA_ERROR(cudaFree(b_dev));
  CHECK_CUDA_ERROR(cudaFree(c_dev));
  CHECK_CUDA_ERROR(cudaFree(temp_dev));

  return res;
}

// ===================== CUTLASS РЕАЛИЗАЦИЯ =====================

/**
 * @brief Обертка для матричного умножения через библиотеку CUTLASS
 * 
 * Использует высокооптимизированные алгоритмы GEMM из библиотеки CUTLASS.
 * Поддерживает Tensor Cores на архитектурах Volta (SM75) и новее.
 * 
 * @param[in] a Указатель на матрицу A в device memory (формат half_t)
 * @param[in] b Указатель на матрицу B в device memory (формат half_t)
 * @param[out] c Указатель на результирующую матрицу C в device memory (float)
 * @param[in] n Размер матриц (n×n)
 * @return cudaSuccess при успешном выполнении, код ошибки CUDA при неудаче
 * 
 * @note Использует row-major layout и вычисление: C = α*A×B + β*C
 * @note В данной реализации α=1.0, β=0.0
 */
cudaError_t CutlassGEMM(const cutlass::half_t* a, const cutlass::half_t* b,
                        float* c, int n) {
  using RowMajor = cutlass::layout::RowMajor;
  using OpClassTensorOp = cutlass::arch::OpClassTensorOp;
  using Sm75 = cutlass::arch::Sm75;
  using CutlassGemm = cutlass::gemm::device::Gemm<cutlass::half_t, RowMajor,
                                                  cutlass::half_t, RowMajor,
                                                  float, RowMajor,
                                                  float, OpClassTensorOp, Sm75>;
  
  CutlassGemm::Arguments args({n, n, n}, {a, n}, {b, n}, {c, n}, {c, n}, {1.0f, 0.0f});
  CutlassGemm gemm_operator;
  cutlass::Status status = gemm_operator(args);
  return status == cutlass::Status::kSuccess ? cudaSuccess : cudaErrorUnknown;
}

/**
 * @brief Реализация матричного умножения с softmax через CUTLASS
 * 
 * Выполняет матричное умножение A×B с использованием библиотеки CUTLASS,
 * затем применяет softmax к результату.
 * 
 * @param[in] matrix Входной вектор, содержащий матрицы A и B
 * @param[in] n Размер матриц (n×n)
 * @return Результирующая матрица после применения softmax к A×B
 * 
 * Алгоритм:
 * 1. Выделение памяти на GPU и копирование данных с CPU
 * 2. Матричное умножение через CUTLASS (CutlassGEMM)
 * 3. Применение softmax через softmax_kernel
 * 4. Копирование результата обратно на CPU
 * 
 * @note Для маленьких матриц (n ≤ 16) используется CPU реализация
 */
std::vector<float> run_cutlass(const std::vector<__half>& matrix, std::size_t n) {
  if (n <= 16) return run_openmp_reference(matrix, n);

  const size_t size = n * n;
  std::vector<float> res(size);
  
  // Debug: buffer for intermediate data
  std::vector<float> gemm_result(size);

  cutlass::half_t *a_dev = nullptr, *b_dev = nullptr;
  float* c_dev = nullptr;
  float* temp_dev = nullptr;

  CHECK_CUDA_ERROR(cudaMalloc(&a_dev, size * sizeof(*a_dev)));
  CHECK_CUDA_ERROR(cudaMalloc(&b_dev, size * sizeof(*b_dev)));
  CHECK_CUDA_ERROR(cudaMalloc(&c_dev, size * sizeof(*c_dev)));
  CHECK_CUDA_ERROR(cudaMalloc(&temp_dev, size * sizeof(*temp_dev)));

  // Initialize with zeros
  CHECK_CUDA_ERROR(cudaMemset(temp_dev, 0, size * sizeof(float)));
  CHECK_CUDA_ERROR(cudaMemset(c_dev, 0, size * sizeof(float)));

  CHECK_CUDA_ERROR(cudaMemcpy(a_dev, matrix.data(), size * sizeof(*matrix.data()),
                              cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(b_dev, matrix.data() + size, size * sizeof(*matrix.data()),
                              cudaMemcpyHostToDevice));

  // 1. Matrix multiplication via CUTLASS
  cudaError_t status = CutlassGEMM(a_dev, b_dev, temp_dev, n);
  CHECK_CUDA_ERROR(status);
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());
  
  // Debug: copy intermediate result
  if (DEBUG_MODE) {
    CHECK_CUDA_ERROR(cudaMemcpy(gemm_result.data(), temp_dev, size * sizeof(float),
                                cudaMemcpyDeviceToHost));
    debug_print_matrix(gemm_result, n, "CUTLASS after multiplication");
    
    // Check value range
    float min_val = gemm_result[0], max_val = gemm_result[0];
    for (const auto& val : gemm_result) {
      min_val = std::min(min_val, val);
      max_val = std::max(max_val, val);
    }
    std::cout << "  CUTLASS multiplication: min=" << min_val << ", max=" << max_val << std::endl;
  }

  // 2. Softmax
  const int softmax_threads = 256;
  dim3 softmax_block(softmax_threads);
  dim3 softmax_grid(n);
  size_t softmax_shared = softmax_threads * sizeof(float);
  softmax_kernel<<<softmax_grid, softmax_block, softmax_shared>>>(temp_dev, c_dev, n);
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  // Copy final result
  CHECK_CUDA_ERROR(cudaMemcpy(res.data(), c_dev, size * sizeof(float),
                              cudaMemcpyDeviceToHost));

  // Debug: show results
  if (DEBUG_MODE) {
    debug_print_matrix(res, n, "CUTLASS final result");
    debug_check_row_sums(res, n, "CUTLASS");
  }

  CHECK_CUDA_ERROR(cudaFree(a_dev));
  CHECK_CUDA_ERROR(cudaFree(b_dev));
  CHECK_CUDA_ERROR(cudaFree(c_dev));
  CHECK_CUDA_ERROR(cudaFree(temp_dev));

  return res;
}

// ===================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ =====================

/**
 * @brief Измеряет время выполнения функции
 * 
 * Замеряет время выполнения переданной функции и сохраняет результат.
 * Используется для бенчмаркинга различных реализаций.
 * 
 * @param[in] work Функция для выполнения (возвращает std::vector<float>)
 * @param[out] result_store Ссылка на вектор для сохранения результата
 * @return Время выполнения в секундах (с плавающей точкой высокой точности)
 */
double measure_seconds(const std::function<std::vector<float>()>& work,
                       std::vector<float>& result_store) {
  const auto start = std::chrono::high_resolution_clock::now();
  result_store = work();
  const auto stop = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double>(stop - start).count();
}

/**
 * @brief Вычисляет максимальную абсолютную разницу между двумя матрицами
 * 
 * Используется для проверки корректности GPU реализаций
 * путем сравнения с эталонной CPU реализацией.
 * 
 * @param[in] baseline Эталонная матрица (обычно CPU реализация)
 * @param[in] candidate Тестируемая матрица (GPU реализация)
 * @return Максимальная абсолютная разница между соответствующими элементами
 * 
 * @throws std::runtime_error если размеры матриц не совпадают
 */
float max_abs_diff(const std::vector<float>& baseline,
                   const std::vector<float>& candidate) {
  if (baseline.size() != candidate.size()) {
    throw std::runtime_error("Matrix sizes don't match!");
  }
  
  float max_diff = 0.0f;
  for (std::size_t i = 0; i < baseline.size(); ++i) {
    float diff = std::abs(baseline[i] - candidate[i]);
    if (diff > max_diff) {
      max_diff = diff;
    }
  }
  return max_diff;
}

/**
 * @struct RunResult
 * @brief Структура для хранения результатов запуска теста
 * 
 * Содержит время выполнения, максимальную разницу с эталоном
 * и флаг успешности выполнения.
 */
struct RunResult {
  double seconds = 0.0;  ///< Время выполнения в секундах
  float diff = 0.0f;     ///< Максимальная разница с эталоном
  bool success = false;  ///< Флаг успешного выполнения
  
  /**
   * @brief Приведение к bool для удобства проверки успешности
   * @return true если тест выполнен успешно, false в противном случае
   */
  explicit operator bool() const noexcept { return success; }
};

/**
 * @brief Форматирует время в строку
 * 
 * Преобразует время в секундах в строку с фиксированной точностью.
 * 
 * @param[in] seconds Время в секундах
 * @return Отформатированная строка с точностью до 2 знаков после запятой
 */
std::string format_time(double seconds) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2) << seconds;
  return oss.str();
}

/**
 * @brief Форматирует разницу между матрицами в строку
 * 
 * Для маленьких значений (<1e-10) возвращает "< 1e-10",
 * для больших - использует научную нотацию.
 * 
 * @param[in] diff Значение разницы
 * @return Отформатированная строка
 */
std::string format_diff(float diff) {
  std::ostringstream oss;
  if (diff < 1e-10) {
    oss << "< 1e-10";
  } else {
    oss << std::scientific << std::setprecision(1) << diff;
  }
  return oss.str();
}

/**
 * @brief Выводит отчет о выполнении теста
 * 
 * Форматирует и выводит информацию о времени выполнения
 * и разнице с эталоном для одного метода.
 * 
 * @param[in] testName Название теста (имя метода)
 * @param[in] result Результаты теста
 */
void print_report(std::string_view testName, const RunResult& result) {
  if (result) {
    std::cout << testName << ": " << format_time(result.seconds)
              << " sec (diff: " << format_diff(result.diff) << ")" << std::endl;
  } else {
    std::cout << testName << ": n/a (diff: n/a)" << std::endl;
  }
}
}  // namespace

// ===================== ОСНОВНАЯ ФУНКЦИЯ =====================

/**
 * @brief Основная функция программы
 * 
 * Управляет выполнением бенчмарка: парсит аргументы, запускает тесты,
 * сравнивает результаты и выводит отчет.
 * 
 * @param[in] argc Количество аргументов командной строки
 * @param[in] argv Массив аргументов командной строки
 * @return EXIT_SUCCESS при успешном выполнении, EXIT_FAILURE при ошибке
 * 
 * Поддерживаемые аргументы:
 * - <n> - размер матрицы (обязательный)
 * - --debug или -d - включение отладочного режима (опциональный)
 */
int main(int argc, char* argv[]) {
  try {
    // Check arguments for --debug flag
    bool debug_mode = false;
    std::size_t n = 0;
    
    for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];
      if (arg == "--debug" || arg == "-d") {
        debug_mode = true;
      } else if (std::all_of(arg.begin(), arg.end(), ::isdigit)) {
        n = static_cast<std::size_t>(std::stoul(arg));
      }
    }
    
    if (n == 0) {
      std::cerr << "Usage: " << argv[0] << " <matrix_size_n> [--debug]\n";
      std::cerr << "  --debug or -d: enable detailed debug output\n";
      return EXIT_FAILURE;
    }
    
    DEBUG_MODE = debug_mode;
    
    if (DEBUG_MODE) {
      std::cout << "Matrix size: " << n << "x" << n << std::endl;
      std::cout << "Debug mode: ENABLED" << std::endl;
    }
    
    // Generate data
    const auto input = make_input_matrix(n);
    
    // Warmup runs for stability
    if (DEBUG_MODE) {
      std::cout << "\n=== Warmup runs ===" << std::endl;
    }
    
    // OpenMP (reference)
    std::vector<float> openmp_result;
    const double openmp_seconds = measure_seconds(
        [&]() { return run_openmp_reference(input, n); }, openmp_result);

    // WMMA
    RunResult wmma_res;
    try {
      std::vector<float> result;
      wmma_res.seconds = measure_seconds([&]() { return run_wmma(input, n); }, result);
      wmma_res.diff = max_abs_diff(openmp_result, result);
      wmma_res.success = true;
      
      // Debug: compare matrices
      if (DEBUG_MODE) {
        debug_compare_matrices(openmp_result, result, n, "OpenMP", "WMMA");
      }
    } catch (const std::exception& ex) {
      std::cerr << "WMMA method failed: " << ex.what() << '\n';
    }

    // CUTLASS
    RunResult cutlass_res;
    try {
      std::vector<float> result;
      cutlass_res.seconds = measure_seconds([&]() { return run_cutlass(input, n); }, result);
      cutlass_res.diff = max_abs_diff(openmp_result, result);
      cutlass_res.success = true;
      
      // Debug: compare matrices
      if (DEBUG_MODE) {
        debug_compare_matrices(openmp_result, result, n, "OpenMP", "CUTLASS");
      }
    } catch (const std::exception& ex) {
      std::cerr << "CUTLASS method failed: " << ex.what() << '\n';
    }

    // Output results
    if (DEBUG_MODE) {
      std::cout << "\n=== Final results ===" << std::endl;
    }
    
    std::cout << "OpenMP: " << format_time(openmp_seconds) << " sec" << std::endl;
    print_report("WMMA", wmma_res);
    print_report("CUTLASS", cutlass_res);
    
    // Final verdict
    if (DEBUG_MODE) {
      std::cout << "\n=== Validation summary ===" << std::endl;
      bool all_good = true;
      
      if (wmma_res.success) {
        if (wmma_res.diff < 1e-4f) {
          std::cout << "WMMA: PASSED (diff < 1e-4)" << std::endl;
        } else if (wmma_res.diff < 1e-2f) {
          std::cout << "WMMA: ACCEPTABLE ACCURACY (diff = " << format_diff(wmma_res.diff) << ")" << std::endl;
          all_good = false;
        } else {
          std::cout << "WMMA: CRITICAL ERROR (diff = " << format_diff(wmma_res.diff) << ")" << std::endl;
          all_good = false;
        }
      }
      
      if (cutlass_res.success) {
        if (cutlass_res.diff < 1e-4f) {
          std::cout << "CUTLASS: PASSED (diff < 1e-4)" << std::endl;
        } else if (cutlass_res.diff < 1e-2f) {
          std::cout << "CUTLASS: ACCEPTABLE ACCURACY (diff = " << format_diff(cutlass_res.diff) << ")" << std::endl;
          all_good = false;
        } else {
          std::cout << "CUTLASS: CRITICAL ERROR (diff = " << format_diff(cutlass_res.diff) << ")" << std::endl;
          all_good = false;
        }
      }
      
      if (all_good) {
        std::cout << "\nALL TESTS PASSED SUCCESSFULLY!" << std::endl;
      } else {
        std::cout << "\nTHERE ARE ACCURACY ISSUES!" << std::endl;
        std::cout << "   Check WMMA/CUTLASS implementation for large matrices." << std::endl;
      }
      
      return all_good ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
    
  } catch (const std::exception& ex) {
    std::cerr << "Error: " << ex.what() << '\n';
  } catch (...) {
    std::cerr << "Unknown error\n";
  }

  return EXIT_FAILURE;
}