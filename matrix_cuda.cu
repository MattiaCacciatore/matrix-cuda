/*
 *  file name: matrix.cu
 *
 *  matrix.cu contains the code that realize matrix multiplication operation in CUDA
 *  
 *  This is a toy program for learning CUDA, some functions are reusable in other project.
 */
#include <iostream>
#include <string>
#include <cmath>          // abs().
#include <chrono>         // To count time elapsed.
#include <cuda_runtime.h> // To get gpu model's name.

#ifdef _WIN32
   #include <windows.h>   // To get cpu model's name.
#elif __unix__
    #include <fstream>    // cpuinfo for Unix.
#endif

using namespace std::chrono;

typedef int matrix_type;
// Tollerance for floating points operations.
const matrix_type mytoll = 0.1;

#define BLOCK_SIZE 256
/* 
It performs product of two matrix (not only square) using GPU Nvidia.

@param &a - GPU device pointer to a m X n matrix (A)
@param &b - GPU device pointer to a n X k matrix (B)
@param &c - GPU device output purpose pointer to a m X k matrix (C) to store the result
@param m  - the size of the A matrix (number of rows)
@param n  - the size of the B matrix (number of columns)
@param k  - the size of the B matrix (number of rows)

Note: grid and block should be configured as:
      dim3 dimGrid((k + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);
      dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
*/
__global__ void gpu_matrix_mult(matrix_type* a, matrix_type* b, matrix_type* c, const int m, const int n, const int k){ 
    const int row = blockIdx.y * blockDim.y + threadIdx.y; 
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    matrix_type sum = 0;

    if(col < k && row < m){
        for(int i = 0; i < n; i++){
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
}
/*
It performs multiplication of two matrix (not only square) using CPU.

@param &a - CPU device pointer to a m X n matrix (A)
@param &b - CPU device pointer to a n X k matrix (B)
@param &c - CPU device output purpose pointer to a m X k matrix (C) to store the result
@param m  - the size of the A matrix (number of rows)
@param n  - the size of the B matrix (number of columns)
@param k  - the size of the B matrix (number of rows)
*/
void cpu_matrix_mult(matrix_type* c_a, matrix_type* c_b, matrix_type* c_c, const int m, const int n, const int k){
    matrix_type sum = 0;
    for (int i = 0; i < m; ++i){
        for (int j = 0; j < k; ++j){
            sum = 0;
            for (int h = 0; h < n; ++h){
                sum += c_a[i * n + h] * c_b[h * k + j];
            }
            c_c[i * k + j] = sum;
        }
    }
}
//---------------------------------------------------------MAIN---------------------------------------------------------------
int main(){
    int m, n, k;
    srand(time(NULL));
    std::cout << "Insert M: ";
    std::cin >> m;
    std::cout << "Insert N: ";
    std::cin >> n;
    std::cout << "Insert K: ";
    std::cin >> k;

    if(m < 1 || n < 1 || k < 1) m = n = k = 1;

#ifdef __unix__
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line, cpu_name;
#endif

    // Count the execution time.
    double gpu_elapsed_time_ms = 0, cpu_elapsed_time_ms = 0;
    high_resolution_clock::time_point start, end;

    std::wstring processor_name;

    matrix_type* cpu_a,* cpu_b,* cpu_c,* gpu_r,* gpu_a,* gpu_b,* gpu_c;

    unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    // Allocate memory in host RAM, gpu_r is used to store CPU result. Allocate memory space on the device (GPU).
    if(cudaMallocHost((void **) &cpu_a, sizeof(matrix_type)*m*n) != cudaSuccess || cudaMallocHost((void **) &cpu_b, sizeof(matrix_type)*n*k) != cudaSuccess
       || cudaMallocHost((void **) &cpu_c, sizeof(matrix_type)*m*k) != cudaSuccess  || cudaMallocHost((void **) &gpu_r, sizeof(matrix_type)*m*k) != cudaSuccess
       || cudaMalloc((void **) &gpu_a, sizeof(matrix_type)*m*n) != cudaSuccess || cudaMalloc((void **) &gpu_b, sizeof(matrix_type)*n*k) != cudaSuccess 
       || cudaMalloc((void **) &gpu_c, sizeof(matrix_type)*m*k) != cudaSuccess)
        goto ErrorAndFree;

    // Random initialization of matrix A and B.
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            //cpu_a[i * n + j] = ((rand() % 256) + (rand() % 2 == 0 ? 0.2 : -0.3)) * (rand() % 2 == 0 ? 1 : -1); // float.
            //cpu_b[i * k + j] = ((rand() % 256) + (rand() % 2 == 0 ? 0.2 : -0.3)) * (rand() % 2 == 0 ? 1 : -1);
            cpu_a[i * n + j] = rand() % 256; // int.
            cpu_b[i * n + j] = rand() % 256;
        }
    }

    // Start to count execution time of GPU version.
    start = high_resolution_clock::now();
    // Copy matrix A and B from host to device memory.
    if(cudaMemcpy(gpu_a, cpu_a, sizeof(matrix_type)*m*n, cudaMemcpyHostToDevice) != cudaSuccess || cudaMemcpy(gpu_b, cpu_b, sizeof(matrix_type)*n*k, cudaMemcpyHostToDevice) != cudaSuccess)
        goto ErrorAndFree;
    // Launch kernel.
    gpu_matrix_mult<<<dimGrid, dimBlock>>>(gpu_a, gpu_b, gpu_c, m, n, k);
    // Transefr results from device to host 
    if(cudaMemcpy(gpu_r, gpu_c, sizeof(matrix_type)*m*k, cudaMemcpyDeviceToHost) != cudaSuccess)
        goto ErrorAndFree;
    // cudaThreadSynchronize is deprecated. Need to set a thread barrier here.
    if(cudaDeviceSynchronize() != cudaSuccess)
        goto ErrorAndFree;
    // Time counting terminate.
    end = high_resolution_clock::now();
    gpu_elapsed_time_ms = duration_cast<milliseconds>(end - start).count();

    // Try to get the gpu model's name. It should be the gpu model's name, take it with a grain of salt and test it.
    cudaDeviceProp gpu_properties;
    cudaGetDeviceProperties(&gpu_properties, 0);
    // Compute time elapse on GPU computing.
    std::cout << "\nTime elapsed on matrix multiplication of " << m << "X" << n << " and " << n << "X" << k << " on\n"
                << gpu_properties.name << "\n(GPU) : ";
    printf("%.3f", (gpu_elapsed_time_ms/1000.0));
    std::cout << " seconds\n\n";
    goto ErrorAndFree;

    // Start to count execution time of CPU version.
    start = high_resolution_clock::now();
    cpu_matrix_mult(cpu_a, cpu_b, cpu_c, m, n, k);
    // Time counting terminate.
    end = high_resolution_clock::now();
	cpu_elapsed_time_ms   = duration_cast<milliseconds>(end - start).count();

    // Try to get the cpu model's name.
#ifdef _WIN32
    SYSTEM_INFO system_info;
    GetSystemInfo(&system_info);

    HKEY hKey;

    if(RegOpenKeyExW(HKEY_LOCAL_MACHINE, L"HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0", 0, KEY_QUERY_VALUE, &hKey) == ERROR_SUCCESS){
        wchar_t data[256];
        DWORD dataSize = sizeof(data);

        if(RegQueryValueExW(hKey, L"ProcessorNameString", nullptr, nullptr, (LPBYTE)data, &dataSize) == ERROR_SUCCESS){
            processor_name = data;
        }

        RegCloseKey(hKey);
    }
#elif __unix__
    for(;std::getline(cpuinfo, line);){
        if(line.find("model name") != std::string::npos){
            cpu_name = line.substr(line.find(":") + 2);
            break;
        }
    }
#endif
    // Compute time elapse on CPU computing.
    std::cout << "Time elapsed on matrix multiplication of " << m << "X" << n << " and " << n << "X" << k << " on\n";
#ifdef _WIN32
    std::wcout << processor_name;
#elif __unix__
    std::cout << cpu_name;
#endif
    printf("\n(CPU) : %.3f", (cpu_elapsed_time_ms/1000.0));
    std::cout << " seconds\n\n";
    // validate results computed by GPU.
    for (int i = 0; i < m; ++i){
        for (int j = 0; j < k; ++j){
            if(gpu_r[i*k + j] - cpu_c[i*k + j] > mytoll){
                std::cout << "Error - gpu_r[" << i << "*" << k << "+" << j << "] : ";
                //printf("%f", gpu_r[i*k + j]); 
                std::cout << " - cpu_c[" << i << "*" << k << "+" << j << "] : ";
                //printf("%f", cpu_c[i*k + j]);
                std::cout << " = " << gpu_r[i*k + j] - cpu_c[i*k + j] << "\n";
                goto ErrorAndFree;
            }
        }
    }

    if(gpu_elapsed_time_ms <= 0) gpu_elapsed_time_ms = 1;
    // Roughly compute speedup.
    std::cout << "GPU performed the multiplication " << (int)(cpu_elapsed_time_ms / gpu_elapsed_time_ms) << " times faster than CPU\n\n";
    // Free memory.
    cudaFree(gpu_a);
    cudaFree(gpu_b);
    cudaFree(gpu_c);
    cudaFreeHost(cpu_a);
    cudaFreeHost(cpu_b);
    cudaFreeHost(cpu_c);
    cudaFreeHost(gpu_r);
    return 0;

ErrorAndFree:
    std::cout << "Something went wrong!\n";
    cudaFree(gpu_a);
    cudaFree(gpu_b);
    cudaFree(gpu_c);
    cudaFreeHost(cpu_a);
    cudaFreeHost(cpu_b);
    cudaFreeHost(cpu_c);
    cudaFreeHost(gpu_r);
    return -1;
}
