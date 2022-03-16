//
// Created by 40169 on 2022/3/15.
//
#include <stdio.h>
const int ROWS = 1;
const int COLS = 3;

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
            c[i*M+j]=sum;
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
            sum+=a[ty*k+i]*b[i*n+tx];
        }
        c[ty*n+tx]=sum;
    }
}

int main()
{
    int total_size = ROWS*COLS*sizeof(float);
    int res_size = COLS*COLS*sizeof(float);
    float *a,*b,*c;
    a = (float*)malloc(total_size);
    b = (float*)malloc(total_size);
    c = (float*)malloc(res_size);
    for(int i=0;i<ROWS*COLS;i++)
    {
        a[i]=2.0;
        b[i]=3.0;
    }
    matrixMulCpuOneDim(a,b,c,COLS,COLS,ROWS);
    for(int i=0;i<COLS;i++)
    {
        for(int j=0;j<COLS;j++)
        {
            printf("%f ",c[i*COLS+j]);
        }
        printf("\n");
    }
    free(a);
    free(b);
    free(c);
    return 0;
}