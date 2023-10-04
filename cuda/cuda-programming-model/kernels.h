#include <sys/time.h>
//#include <cuda_runtime.h>
#include <stdbool.h>
#include <stdio.h>
#include <lib.h>

// p108.cu

__global__ void warmup(int * g_idata, int *g_odata, unsigned int n);
__global__ void reduceNeighboredLess(int * g_idata, int *g_odata, unsigned int n);
__global__ void reduceInterleaved(int * g_idata, int *g_odata, unsigned int n);

__global__ void warmingup(float * c);
__global__ void mathKernel1(float * c);
__global__ void mathKernel2(float * c);
__global__ void mathKernel3(float * c);
__global__ void mathKernel4(float * c);
__global__ void sumMatrixOnGPU2D(float *MatA, float *MatB, float *MatC, int nx, int ny);
