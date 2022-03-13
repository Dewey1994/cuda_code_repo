//
// Created by 40169 on 2022/3/13.
//

#pragma once
#include <stdio.h>


#define CHECK(call)\
do\
{\
    const cudaError_t error_code = call;\
    if(error_code != cudaSuccess)\
    {\
        printf("CUDA ERROR:\n");\
        printf("    FILE:    %s\n",__FILE__);\
        printf("    LINE:    %d\n",__LINE__);\
        printf("    Error code:    %d\n",error_code);\
        printf("    Error text:    %s\n",cudaGetErrorString(error_code));\
        exit(1);\
    }\
}\
while(0)
