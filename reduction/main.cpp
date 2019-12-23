#include <iostream>
#include <vector>
#include <random>
#include <algorithm>

#include <chrono>

#include <cuda_runtime.h>

#include "helper_cuda.h"

#include "reduction_cpu.hpp"
#include "gpu/reduction_gpu.h"


template<class T>
std::vector<double> benchmark_cpu(
    const std::vector<T>& array,
    std::vector<T>& sum_golds,
    const size_t& num_loop=100
    )
{
    std::vector<double> calc_times;

    for(size_t i = 0; i < num_loop; ++i){
        const auto cpu_start = std::chrono::system_clock::now();
        const auto sum_gold = cpu::reduction_gold(array);
        const auto cpu_end = std::chrono::system_clock::now();

        const auto cpu_dur = cpu_end - cpu_start;

        const auto cpu_dur_ms = std::chrono::duration_cast<std::chrono::nanoseconds>(cpu_dur).count() / (1000.*1000.);

        sum_golds.push_back(sum_gold);
        calc_times.push_back(cpu_dur_ms);
    }

    return calc_times;
}


template<class T>
std::vector<double> benchmark_gpu(
    const std::vector<T>& array,
    std::vector<T>& sum_results,
    const size_t& num_loop=100
    )
{
    const size_t bytes = array.size() * sizeof(T);

    // GPU params setup
    int num_blocks = 256;
    int num_threads = 32;
    // get

    // GPU Global Memory setup
    T* d_iarray;
    T* d_oarray;

    checkCudaErrors(cudaMalloc((void **)&d_iarray, bytes));
    checkCudaErrors(cudaMalloc((void **)&d_oarray, num_blocks*sizeof(T)));

    std::vector<double> calc_times;

    for(size_t i = 0; i < num_loop; ++i){
        const auto cpu_start = std::chrono::system_clock::now();

        checkCudaErrors(cudaMemcpy(d_iarray, &array[0], bytes, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_oarray, &array[0], num_blocks*sizeof(T), cudaMemcpyHostToDevice));

        reduction_gpu(bytes, d_iarray, d_oarray, num_threads, num_blocks);

        const auto cpu_end = std::chrono::system_clock::now();

        const auto cpu_dur = cpu_end - cpu_start;

        const auto cpu_dur_ms = std::chrono::duration_cast<std::chrono::nanoseconds>(cpu_dur).count() / (1000.*1000.);

        int sum_result = 0;

        sum_results.push_back(sum_result);
        calc_times.push_back(cpu_dur_ms);
    }

    checkCudaErrors(cudaFree(d_iarray));
    checkCudaErrors(cudaFree(d_oarray));

    return calc_times;
}


double calc_time_median(std::vector<double> times)
{
    std::sort(std::begin(times), std::end(times));
    
    const size_t median_idx = times.size() / 2;
    const double median = (times.size() % 2 == 0 ? static_cast<double>(times[median_idx] + times[median_idx - 1]) / 2 : times[median_idx]);

    return median;
}


int main()
{
    using T = int;

    const size_t max_array_size = 1 << 24;

    std::cout << "Max array size: " << max_array_size << std::endl;

    std::random_device rnd;

    std::vector<T> target_array(max_array_size);
    for(auto& v : target_array){
        v = (T)(rnd() & 0xFF);
    }

    std::vector<T> sum_golds;
    sum_golds.reserve(100);
    const auto cpu_times = benchmark_cpu(target_array, sum_golds, 100);
    const auto sum_gold = sum_golds[0];
    for(size_t i = 1; i < sum_golds.size(); ++i){
        if(std::abs(sum_golds[i] - sum_gold) > 1.0e-8){
            std::cout << "ERROR: sum_gold of CPU." << std::endl;
            return -1;
        }
    }

    std::cout << "CPU time(Median) " << calc_time_median(cpu_times) << " ms" << std::endl;

    // GPU Benchmark
    cudaSetDevice(0);

    size_t bytes = max_array_size * sizeof(T);

    // Call benchmark_gpu fn

    // GPU params setup
    int num_blocks = 256;
    int num_threads = 32;
    // get

    // GPU Global Memory setup
    T* d_iarray;
    T* d_oarray;

    checkCudaErrors(cudaMalloc((void **)&d_iarray, bytes));
    checkCudaErrors(cudaMalloc((void **)&d_oarray, num_blocks*sizeof(T)));

    checkCudaErrors(cudaMemcpy(d_iarray, &target_array[0], bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_oarray, &target_array[0], num_blocks*sizeof(T), cudaMemcpyHostToDevice));

    reduction_gpu(bytes, d_iarray, d_oarray, num_threads, num_blocks);

    checkCudaErrors(cudaFree(d_iarray));

    cudaDeviceReset();
    return 0;
}