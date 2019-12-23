#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>

#include <chrono>

#include <cuda_runtime.h>

#include "helper_cuda.h"

#include "reduction_cpu.hpp"
#include "gpu/reduction_gpu.h"



unsigned int nextPow2(unsigned int x)
{
    --x;

    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;

    return ++x;
}


void getNumBlocksAndThreads(int whichKernel, int n, int maxBlocks, int maxThreads, int& blocks, int& threads)
{

    //get device capability, to avoid block/grid size exceed the upper bound
    cudaDeviceProp prop;
    int device;
    checkCudaErrors(cudaGetDevice(&device));
    checkCudaErrors(cudaGetDeviceProperties(&prop, device));

    if (whichKernel < 3)
    {
        threads = (n < maxThreads) ? nextPow2(n) : maxThreads;
        blocks = (n + threads - 1) / threads;
    }
    else
    {
        threads = (n < maxThreads * 2) ? nextPow2((n + 1) / 2) : maxThreads;
        blocks = (n + (threads * 2 - 1)) / (threads * 2);
    }

    if ((float)threads * blocks > (float)prop.maxGridSize[0] * prop.maxThreadsPerBlock)
    {
        printf("n is too large, please choose a smaller number!\n");
    }

    if (blocks > prop.maxGridSize[0])
    {
        printf("Grid size <%d> exceeds the device capability <%d>, set block size as %d (original %d)\n",
            blocks, prop.maxGridSize[0], threads * 2, threads);

        blocks /= 2;
        threads *= 2;
    }

    if (whichKernel == 6)
    {
        blocks = std::min(maxBlocks, blocks);
    }
}


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
    const size_t i_bytes = array.size() * sizeof(T);

    // GPU params setup
    int num_blocks = 0;
    int num_threads = 0;
    getNumBlocksAndThreads(0, array.size(), 64, 256, num_blocks, num_threads);

    const size_t o_bytes = num_blocks * sizeof(T);

    // GPU Global Memory setup
    T* d_iarray;
    T* d_oarray;

    checkCudaErrors(cudaMalloc((void **)&d_iarray, i_bytes));
    checkCudaErrors(cudaMalloc((void **)&d_oarray, o_bytes));

    std::vector<double> calc_times;

    for(size_t i = 0; i < num_loop; ++i){
        std::vector<T> block_res(num_blocks, 0);

        const auto start_time = std::chrono::system_clock::now();

        checkCudaErrors(cudaMemcpy(d_iarray, &array[0], i_bytes, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_oarray, &array[0], o_bytes, cudaMemcpyHostToDevice));

        reduction_gpu(i_bytes, d_iarray, d_oarray, num_threads, num_blocks);

        checkCudaErrors(cudaMemcpy(&block_res[0], d_oarray, o_bytes, cudaMemcpyDeviceToHost));

        const auto end_time = std::chrono::system_clock::now();

        const auto dur_time = end_time - start_time;

        const auto dur_time_ms = std::chrono::duration_cast<std::chrono::nanoseconds>(dur_time).count() / (1000.*1000.);

        T sum_result = std::accumulate(std::begin(block_res), std::end(block_res), (T)0);

        sum_results.push_back(sum_result);
        calc_times.push_back(dur_time_ms);
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

    const size_t num_loop = 100;

    const size_t max_array_size = 1 << 24;
    //const size_t max_array_size = 32;

    std::cout << "Max array size: " << max_array_size << std::endl;

    std::random_device rnd;

    std::vector<T> target_array(max_array_size);
    for(auto& v : target_array){
        v = (T)(rnd() & 0xFF);
        //std::cout << v << ",";
    }
    //std::cout << std::endl;

    std::vector<T> sum_golds;
    sum_golds.reserve(num_loop);
    const auto cpu_times = benchmark_cpu(target_array, sum_golds, num_loop);
    const auto sum_gold = sum_golds[0];
    for(const auto& v : sum_golds){
        if(std::abs(v - sum_gold) > std::numeric_limits<T>::epsilon()){
            std::cout << "ERROR: sum_gold of CPU." << std::endl;
            return -1;
        }
    }
    std::cout << "CPU result is " << sum_gold << std::endl;

    std::cout << "CPU time(Median) " << calc_time_median(cpu_times) << " ms" << std::endl;

    // GPU Benchmark
    cudaSetDevice(0);

    size_t cound_missmatches = 0;

    std::vector<T> sum_results;
    sum_results.reserve(num_loop);

    const auto gpu_times = benchmark_gpu(target_array, sum_results, num_loop);

    for(const auto& v : sum_results){
        if(std::abs(v - sum_gold) > std::numeric_limits<T>::epsilon()){
            ++cound_missmatches;
        }
    }

    std::cout << "GPU result is " << sum_results[0] << std::endl;

    std::cout << "GPU: The number of mismatch results is " << cound_missmatches << std::endl;
    std::cout << "GPU time(Median) " << calc_time_median(gpu_times) << " ms" << std::endl;

    cudaDeviceReset();
    return 0;
}