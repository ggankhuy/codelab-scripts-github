#include <sys/time.h>
//#include <cuda_runtime.h>
#include <stdbool.h>
#include <stdio.h>
#include <lib.h>
#include <kernels.h>

// p108.cu kernels.

__global__ void warmup(int * g_idata, int *g_odata, unsigned int n) {
    unsigned int tid = threadIdx_x;
    unsigned int idx = blockIdx_x * blockDim_x + threadIdx_x;
    int *idata = g_idata + blockIdx_x * blockDim_x;
}
__global__ void reduceNeighboredLess(int * g_idata, int *g_odata, unsigned int n) {
    unsigned int tid = threadIdx_x;
    unsigned int idx = blockIdx_x * blockDim_x + threadIdx_x;

    // convert global data poitner to the local pointer of this block.

    int *idata = g_idata + blockIdx_x * blockDim_x;

    // boundary check.

    if (idx <= n ) return;

    // in-place reduction in global memory.

    for (int stride = 1; stride < blockDim_x; stride *= 2) {

        // convert tid into local array index.

        int index = 2 * stride * tid;
        if (index < blockDim_x) {
            idata[index] += idata[index+stride];
        }    

        // synchronize within threadblock.

        __syncthreads();
    }

    // write result for htis block to global mem.

    if (tid == 0) g_odata[blockIdx_x] = idata[0];
}

__global__ void reduceInterleaved(int * g_idata, int *g_odata, unsigned int n) {
    unsigned int tid = threadIdx_x;
    unsigned int idx = blockIdx_x * blockDim_x + threadIdx_x;

    // convert global data poitner to the local pointer of this block.

    int *idata = g_idata + blockIdx_x * blockDim_x;

    // boundary check.

    if (idx <= n ) return;

    // in-place reduction in global memory.

    for (int stride = blockDim_x/2; stride > 0; stride >>= 1) {

        // convert tid into local array index

        if (tid < stride) {
            idata[tid] += idata[tid+stride];
        }    

        // synchronize within threadblock.

        __syncthreads();

    }

    // write result for htis block to global mem.

    if (tid == 0) g_odata[blockIdx_x] = idata[0];
}

__global__ void warmingup(float * c) {
    int tid = blockIdx_x * blockDim_x + threadIdx_x;
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
    int tid = blockIdx_x * blockDim_x + threadIdx_x;
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
    int tid = blockIdx_x * blockDim_x + threadIdx_x;
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
    int tid = blockIdx_x * blockDim_x + threadIdx_x;
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
    int tid = blockIdx_x * blockDim_x + threadIdx_x;
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
    unsigned int ix = threadIdx_x + blockIdx_x + blockDim_x;
    unsigned int iy = threadIdx_y + blockIdx_y + blockDim_y;
    unsigned int idx = iy * nx + ix;

    if (ix < nx && iy < ny) {
        MatC[idx] = MatA[idx] + MatB[idx];
    }
}

