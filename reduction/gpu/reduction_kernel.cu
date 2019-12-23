// #include <stdio>
#include <cuda_runtime.h>
// #include <helper_cuda.h>
// #include <helper_functions.h>


__global__
void reduction_lvl0(int *g_iarray, int *g_oarray, unsigned long long array_size)
{
    extern __shared__ int s_data[];

    unsigned int thread_id = threadIdx.x;
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

    s_data[thread_id] = idx < array_size ? g_iarray[idx] : 0;

    __syncthreads();

    // Add stride 2
    for(unsigned int i = 1; i < blockDim.x; i *= 2){
        if(thread_id % (2*i) == 0){
            s_data[thread_id] += s_data[thread_id + i];
        }
        __syncthreads();
    }

    // partial sum of a block
    if(thread_id == 0){
        g_oarray[blockIdx.x] = s_data[0];
    }
}


void reduction_gpu(unsigned long long array_size, int* d_iarray, int* d_oarray, const int threads, const int blocks)
{
    dim3 dim_block(threads, 1, 1);
    dim3 dim_grid(blocks, 1, 1);

    int shared_mem_size = threads <= 32 ? 2*threads*sizeof(int) : threads*sizeof(int);

    reduction_lvl0<<<dim_grid, dim_block, shared_mem_size>>>(d_iarray, d_oarray, array_size);

}