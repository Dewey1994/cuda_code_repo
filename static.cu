//
// Created by 40169 on 2022/3/14.
//
#include "include/error.cuh"

__device__ int d_x = 1;
__device__ int d_y[2];

void __global__ my_kernel()
{
    d_y[0]+=d_x;
    d_y[1]+=d_x;
}

int main()
{
    int h_y[2] = {10,20};
    CHECK(cudaMemcpyToSymbol(d_y,h_y,sizeof(int)*2));
    my_kernel<<<1,1>>>();
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpyFromSymbol(h_y,d_y, sizeof(int)*2));
    return 0;
}


