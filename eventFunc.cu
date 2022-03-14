//
// Created by 40169 on 2022/3/14.
//

#include "include/error.cuh"

__host__ void eventF()
{
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));
    cudaEventQuery(start);

    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    float elapsed_time;
    CHECK(cudaEventElapsedTime(&elapsed_time,start,stop));
    printf("Time = %g ms. \n",elapsed_time);
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
}
