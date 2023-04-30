#include <sys/time.h>
//#include <cuda_runtime.h>
#include <stdbool.h>
#include <stdio.h>
#include <lib.h>
#include <kernels.h>

__global__ void warmingup(float * c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a,b;
    a = b = 0.0f;
    
    if (tid % 2 == 0 ) {
        a = 100.0f;
    } else {
        b = 200.0f;        
    }
    c[tid] = a + b;
} 

__global__ void mathKernel1(float * c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a,b;
    a = b = 0.0f;
    
    if (tid % 2 == 0 ) {
        a = 100.0f;
    } else {
        b = 200.0f;        
    }
    c[tid] = a + b;
} 

__global__ void mathKernel2(float * c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a,b;
    a = b = 0.0f;
    
    if ((tid / warpSize ) % 2  == 0 ) {
        a = 100.0f;
    } else {
        b = 200.0f;        
    }
    c[tid] = a + b;
} 

__global__ void mathKernel3(float * c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a,b;
    a = b = 0.0f;
    
    if (tid % 2 == 0 ) {
        a = 100.0f;
    } else {
        b = 200.0f;        
    }
    c[tid] = a + b;
} 
__global__ void mathKernel4(float * c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a,b;
    a = b = 0.0f;
    
    if (tid % 2 == 0 ) {
        a = 100.0f;
    } else {
        b = 200.0f;        
    }
    c[tid] =a + b;
} 

__global__ void sumMatrixOnGPU2D(float *MatA, float *MatB, float *MatC, int nx, int ny) {
    unsigned int ix = threadIdx.x + blockIdx.x + blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y + blockDim.y;
    unsigned int idx = iy * nx + ix;

    if (ix < nx && iy < ny) {
        MatC[idx] = MatA[idx] + MatB[idx];
    }
}

