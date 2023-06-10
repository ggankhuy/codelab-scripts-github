#include <sys/time.h>
//#include <cuda_runtime.h>
#include <stdbool.h>
#include <stdio.h>
#include <lib.h>

// p108.cu

__global__ void warmup(int * g_idata, int *g_odata, unsigned int n);
__global__ void reduceNeighbored(int * g_idata, int *g_odata, unsigned int n);
__global__ void reduceNeighboredLess(int * g_idata, int *g_odata, unsigned int n);
__global__ void reduceNeighboredInterleaved(int * g_idata, int *g_odata, unsigned int n);
__global__ void reduceGmem(int * g_idata, int *g_odata, unsigned int n);

__global__ void warmingup(float * c);
__global__ void mathKernel1(float * c);
__global__ void mathKernel2(float * c);
__global__ void mathKernel3(float * c);
__global__ void mathKernel4(float * c);

__global__ void sumArraysOnGPU(float *A, float *B, float*C, const int N);
__global__ void sumMatrixOnGPU2D(float *MatA, float *MatB, float *MatC, int nx, int ny);

__global__ void copyRow(float * out, float * in, const int nx, const int ny);
__global__ void copyCol(float * out, float * in, const int nx, const int ny);

__global__ void transposeNaiveRow(float * out, float * in, const int nx, const int ny);
__global__ void transposeNaiveCol(float * out, float * in, const int nx, const int ny);

__global__ void warmup(float * out, float * in, const int nx, const int ny);
__global__ void warmup(float *A, float *B, float *C, const int n, int offset);
__global__ void readOffset(float *A, float *B, float *C, const int n, int offset);

#ifndef P220
#define P220
#define BDIMX 32
#define BDIMY 32
#define IPAD  1
#endif

__global__ void setRowReadRow (int *out);
__global__ void setColReadCol (int *out);
__global__ void setRowReadCol(int *out);
__global__ void setRowReadColDyn(int *out);
__global__ void setRowReadColPad(int *out);
__global__ void setRowReadColDynPad(int *out);

__global__ void transposeSmem(float * out, float * in, int nx, int ny);
__global__ void reduceUnrollWarp8 (int *g_idata, int *g_odata, unsigned int n);
__global__ void reduceCompleteUnrollWarp8 (int *g_idata, int *g_odata, unsigned int n);
__global__ void reduceCompleteUnroll(int *g_idata, int *g_odata, unsigned int n);
