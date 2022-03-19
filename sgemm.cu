//
// Created by 40169 on 2022/3/15.
//
#include <stdio.h>
#include "include/error.cuh"

const int M = 4;
const int K = 4;
const int N = 4;
const int TILE_DIM = 2;
const int TILE_SIZE = 2;

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
    int tx = blockDim.x*blockIdx.x+threadIdx.x;  //tx==rol's idx
    int ty = blockDim.y*blockIdx.y+threadIdx.y;  // ty==col's idx
    if(ty<m && tx<n)
    {
        float sum = 0;
        for(int i=0;i<k;i++)
        {
            sum+=a[ty*k+i]*b[i*n+tx];  //a[i][k]  b[k][j]  x[x1][x2]  == x1*width+x2
        }
        c[ty*n+tx]=sum;
    }
}


__global__ void matrixMulSmem(const float *a, const float *b, float *c,const int m,const int n, const int k)
{   // mxk * kxn
    __shared__ float smem_a[TILE_DIM][TILE_DIM];
    __shared__ float smem_b[TILE_DIM][TILE_DIM];
    float sum = 0.0;
    int tx = threadIdx.x; int ty = threadIdx.y;
    int bx = blockIdx.x; int by = blockIdx.y;
    int row = by*TILE_DIM+ty;
    int col = bx*TILE_DIM+tx;
    for(int i=0;i<M/TILE_SIZE;i++)
    {
        //smem_a[ty][tx] = a[row][i*TILE_SIZE+tx];
        smem_a[ty][tx]=a[i*TILE_SIZE+tx+row*k];
        //smem_b[ty][tx] = b[ty+TILE_SIZE*i][col];
        smem_b[ty][tx]=b[col+N*(ty+TILE_SIZE*i)];
        __syncthreads();
        for(int j=0;j<TILE_SIZE;j++)
        {
            sum+=smem_a[ty][j]*smem_b[j][tx];
            __syncthreads();
        }
    }
    c[row*K+col]=sum;
}


template <typename T>
__global__ void matmul_Tiling(T *A, T *B, T *C, int M, int K, int N) {
    /* Basic tiling implementation of matrix multiplication.
     * Based on a more mathematically reasonable indexing method.
     */
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    __shared__ T As[TILE_SIZE][TILE_SIZE];
    __shared__ T Bs[TILE_SIZE][TILE_SIZE];

    int aBegin = K * TILE_SIZE * by;
    int aEnd = aBegin + K - 1;
    int aStep = TILE_SIZE;

    int bBegin = TILE_SIZE * bx;
    int bStep = TILE_SIZE * N;

    T Csub = 0;

    for (int i = aBegin, j = bBegin; i <= aEnd; i += aStep, j += bStep) {
        As[ty][tx] = A[i + K * ty + tx];
        Bs[tx][ty] = B[j + N * tx + ty];

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            Csub += As[ty][k]*Bs[k][tx];
        }

        __syncthreads();
    }
    int cIdx = N * TILE_SIZE * by + TILE_SIZE * bx;
    C[cIdx + N * ty + tx] = Csub;
}


__global__ void sharedABMultiply(float *a, float* b, float *c,
                                 int N)
{
    __shared__ float aTile[TILE_DIM][TILE_DIM],
            bTile[TILE_DIM][TILE_DIM];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    for(int q=0;q<N/TILE_DIM;q++)
    {
        aTile[threadIdx.y][threadIdx.x] = a[row*N+col];
        bTile[threadIdx.y][threadIdx.x] = b[row*N+col];
        __syncthreads();
        for (int i = 0; i < TILE_DIM; i++) {
            sum += aTile[threadIdx.y][i]* bTile[i][threadIdx.x];
        }
    }
    c[row*N+col] = sum;
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
    a[2]=4;
    for(int i=0;i<K*N;i++)
    {
        b[i]=3.0;
    }
    b[1]=5;
    CHECK(cudaMemcpy(d_a,a,total_size,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b,b,total_size1,cudaMemcpyHostToDevice));

//    matrixMulCpuOneDim(a,b,c,M,N,K);
    matrixMulSmem<<<blocks,threads>>>(d_a,d_b,d_c,M,K,N);
//    sharedABMultiply<<<blocks,threads>>>(d_a,d_b,d_c,K);
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