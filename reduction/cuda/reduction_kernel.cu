#include <stdio>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>


__global__
void reduction_lvl0(int *g_iarray, int *g_oarray, unsigned long long array_size)
{
    unsigned int thread_id = threadIdx.x;
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

    int *s_data = idx < array_size : g_iarray[idx] : 0

    __syncthreads();
    // understand

    // Add stride 2
    for(unsigned int i = 1; i < blockDim.x; i *= 2){
        if(thread_id % (2*i) == 0){
            s_data[thread_id] += s_data[thread_id + i];
        }
        __syncthreads();
    }

    // partial sum of a block
    if(thread_id == 0){
        g_oarray[blockIdx] = s_data[0];
    }
}


__host__
void reduction_gpu(unsigned long long array_size, int *d_iarray, int *d_oarray,
int threads=256, int blocks=32)
{
    // int threads = 256;
    // int blocks = 32;

    dim3 dim_block(threads, 1, 1);
    dim3 dim_grid(blocks, 1, 1);

    int shared_mem_size = threads*sizeof(int);

    reduction_lvl0<<<dim_grid, dim_block>>>(d_iarray, d_oarray, array_size);

}