
#include "reduction_cpu.hpp"

namespace cpu
{
template<class T>
T reduction_gold(const std::vector<T>& array)
{
    T sum = (T)0;
    T coeff = (T)0;

    for(const auto& v : array){
        T y = v - coeff;
        T t = sum + y;
        coeff = (t - sum) - y;
        sum = t;
    }

    return sum;
}

template int reduction_gold<int>(const std::vector<int>&);
template float reduction_gold<float>(const std::vector<float>&);

}