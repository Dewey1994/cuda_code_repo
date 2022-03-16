//
// Created by 40169 on 2022/3/16.
//
#define DIM 1024
__global__ void reduceGmem(int *g_idata, int *g_odata, unsigned int n)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockDim.x*blockIdx.x+threadIdx.x;
    //boundary check
    if(tid >= n) return;
    int *idata = g_idata+blockDim.x*blockIdx.x;
    if(blockDim.x>=1024 && tid<512)
        idata[tid]+=idata[tid+512];
    __syncthreads();
    if(blockDim.x>=512 && tid<256)
        idata[tid]+=idata[tid+256];
    __syncthreads();
    if(blockDim.x>=256 && tid<128)
        idata[tid]+=idata[tid+128];
    __syncthreads();
    if(blockDim.x>=128 && tid<64)
        idata[tid]+=idata[tid+64];
    __syncthreads();
    if(tid<32)
    {
        volatile int *vmem = idata;
        vmem[tid]+=vmem[tid+32];
        vmem[tid]+=vmem[tid+16];
        vmem[tid]+=vmem[tid+8];
        vmem[tid]+=vmem[tid+4];
        vmem[tid]+=vmem[tid+2];
        vmem[tid]+=vmem[tid+1];
    }
    if(tid==0)
        g_odata[blockIdx.x]=idata[0];
}

__global__ void reduceSmem(int *g_idata, int *g_odata, unsigned int n)
{
    __shared__ int smem[DIM];
    unsigned int tid = threadIdx.x;

    if(tid>=n) return;
    int *idata = g_idata+blockDim.x+blockIdx.x;
    smem[tid]=idata[tid];
    __syncthreads();

    if(blockDim.x>=1024 && tid<512)
        smem[tid]+=smem[tid+512];
    __syncthreads();
    if(blockDim.x>=512 && tid<256)
        smem[tid]+=smem[tid+256];
    __syncthreads();
    if(blockDim.x>=256 && tid<128)
        smem[tid]+=smem[tid+128];
    __syncthreads();
    if(blockDim.x>=128 && tid<64)
        smem[tid]+=smem[tid+64];
    __syncthreads();

    if(tid<32)
    {
        volatile int *vsmem = smem;
        vsmem[tid]+=vsmem[tid+32];
        vsmem[tid]+=vsmem[tid+16];
        vsmem[tid]+=vsmem[tid+8];
        vsmem[tid]+=vsmem[tid+4];
        vsmem[tid]+=vsmem[tid+2];
        vsmem[tid]+=vsmem[tid+1];
    }

    if(tid==0)
        g_odata[blockIdx.x] = smem[0];
}
int main()
{

}