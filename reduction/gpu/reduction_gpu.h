#pragma once


void reduction_gpu(unsigned long long array_size, int* d_iarray, int* d_oarray, const int threads, const int blocks);