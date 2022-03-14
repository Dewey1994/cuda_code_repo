//
// Created by 40169 on 2022/3/12.
//
#include <math.h>
#include <stdio.h>
#include "include/error.cuh"

const double EPSLION = 1e-15;
const double a = 1.23;
const double b = 2.34;
const double c = 3.57;
void __global__ add(const double *x, const double *y, double *z);
void check(const double *z, const int N);

int main()
{
    const int N = 1e8;
    const int M = sizeof(double)*N;
    double *h_x = (double*)malloc(M);
    double *h_y = (double*)malloc(M);
    double *h_z = (double*)malloc(M);

    for(int i=0;i<N;i++)
    {
        h_x[i]=a;
        h_y[i]=b;
    }
    double *d_x,*d_y,*d_z;
    cudaMalloc((void **) &d_x,M);
    cudaMalloc((void **) &d_y,M);
    cudaMalloc((void **) &d_z,M);
    CHECK(cudaMemcpy(d_x,h_x,M,cudaMemcpyDeviceToHost));
    cudaMemcpy(d_y,h_y,M,cudaMemcpyHostToDevice);
    const int block_size = 128;
    const int grid_size = N/block_size;
    add<<<grid_size,block_size>>>(d_x,d_y,d_z);
    cudaMemcpy(h_z,d_z,M,cudaMemcpyDeviceToHost);
    check(h_z,N);
    free(h_x);
    free(h_y);
    free(h_z);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    return 0;
}

void __global__ add(const double *x, const double *y, double *z)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    z[idx]=x[idx]+y[idx];
}

void check(const double *z, const int N)
{
    bool is_equal = true;
    for(int i=0;i<N;i++)
    {
        if(fabs(z[i]-c)>EPSLION)
            is_equal = false;
    }
    printf("%s\n",is_equal? "No errors":"has errors");
}

