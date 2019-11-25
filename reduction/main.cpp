#include <iostream>
#include <vector>
#include <random>
#include <algorithm>

#include <chrono>

#include "reduction_cpu.hpp"

template<class T>
std::vector<double> cpu_bench(const std::vector<T>& array, const size_t& num_loop=100)
{
    std::vector<double> calc_times;

    for(size_t i = 0; i < num_loop; ++i){
        const auto cpu_start = std::chrono::system_clock::now();
        const auto sum_gold = cpu::reduction_gold(array);
        const auto cpu_end = std::chrono::system_clock::now();

        const auto cpu_dur = cpu_end - cpu_start;

        const auto cpu_dur_ms = std::chrono::duration_cast<std::chrono::nanoseconds>(cpu_dur).count() / (1000.*1000.);

        calc_times.push_back(cpu_dur_ms);
    }

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
        // std::cout << v << ",";
    }
    // std::cout << std::endl;

    // const auto cpu_start = std::chrono::system_clock::now();
    // const auto sum_gold = cpu::reduction_gold(target_array);
    // const auto cpu_end = std::chrono::system_clock::now();

    // const auto cpu_dur = cpu_end - cpu_start;

    // std::cout << "CPU Elapsed time " << std::chrono::duration_cast<std::chrono::nanoseconds>(cpu_dur).count() / (1000.*1000.) << " ms: ";
    // std::cout << sum_gold << std::endl;

    const auto cpu_times = cpu_bench(target_array, 100);

    std::cout << "CPU time(Median) " << calc_time_median(cpu_times) << " ms" << std::endl;

    return 0;
}