//
// Created by 40169 on 2022/3/15.
//
#include <stdio.h>
#include "include/error.cuh"

const int M = 6;
const int K = 1;
const int N = 3;
const int TILE_DIM = 3;

void matrixMulCpuOneDim(const float *a, const float *b, float *c,int M,int N, int K)
{
    for(int i=0;i<M;i++)
    {
        for(int j=0;j<N;j++)
        {
            float sum = 0.0;
            for(int k=0;k<K;k++)
            {
                sum+=a[i*K+k]*b[k*N+j];
//                c[i][j] = a[i][k]+b[k][j];
            }
            c[i*N+j]=sum;
        }
    }
}
__global__ void matrixMul(const float *a, const float *b, float *c,int m,int n, int k)
{   // mxk * kxn
    int tx = blockDim.x*blockIdx.x+threadIdx.x;
    int ty = blockDim.y*blockIdx.y+threadIdx.y;
    if(ty<m && tx<n)
    {
        float sum = 0;
        for(int i=0;i<k;i++)
        {
            sum+=a[ty*k+i]*b[tx*k+i];
        }
        c[ty*n+tx]=sum;
    }
}


__global__ void matrixMul1(const float *a, const float *b, float *c,int m,int n, int k)
{   // mxk * kxn
    int tx = blockDim.x*blockIdx.x+threadIdx.x;
    int ty = blockDim.y*blockIdx.y+threadIdx.y;
    if(ty<m && tx<n)
    {
        float sum = 0;
        for(int i=0;i<k;i++)
        {
            sum+=a[ty*k+i]*b[i*n+tx];
        }
        c[ty*n+tx]=sum;
    }
}


__global__ void matrixMulSmem(const float *a, const float *b, float *c,const int m,const int n, const int k)
{   // mxk * kxn
    __shared__ float smem_a[TILE_DIM][TILE_DIM];
    __shared__ float smem_b[TILE_DIM][TILE_DIM];
    int cols = blockDim.x*blockIdx.x+threadIdx.x;
    int rows = blockDim.y*blockIdx.y+threadIdx.y;
    smem_a[threadIdx.y][threadIdx.x] = a[rows*TILE_DIM+threadIdx.x];

}

int main()
{
    int total_size = M*K*sizeof(float);
    int total_size1 = N*K*sizeof(float);
    int res_size = M*N*sizeof(float);
    dim3 threads(TILE_DIM,TILE_DIM);
    dim3 blocks(N/TILE_DIM,M/TILE_DIM);
    float *a,*b,*c;
    a = (float*)malloc(total_size);
    b = (float*)malloc(total_size1);
    c = (float*)malloc(res_size);
    float *d_a,*d_b,*d_c;
    CHECK(cudaMalloc((void**)&d_a,total_size));
    CHECK(cudaMalloc((void**)&d_b,total_size1));
    CHECK(cudaMalloc((void**)&d_c,res_size));
    for(int i=0;i<M*K;i++)
    {
        a[i]=2.0;
    }
    for(int i=0;i<K*N;i++)
    {
        b[i]=3.0;
    }
    CHECK(cudaMemcpy(d_a,a,total_size,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b,b,total_size1,cudaMemcpyHostToDevice));

//    matrixMulCpuOneDim(a,b,c,M,N,K);
     matrixMul<<<blocks,threads>>>(d_a,d_b,d_c,M,N,K);
     // blocks threads 设置需要(y,x)反着来，代码按正常逻辑写
    CHECK(cudaMemcpy(c,d_c,res_size,cudaMemcpyDeviceToHost));
    for(int i=0;i<M;i++)
    {
        for(int j=0;j<N;j++)
        {
            printf("%f ",c[i*N+j]);
        }
        printf("\n");
    }
    free(a);
    free(b);
    free(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}