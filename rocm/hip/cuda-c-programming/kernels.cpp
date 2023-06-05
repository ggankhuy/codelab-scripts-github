#include <sys/time.h>
#include <hip/hip_runtime.h>
#include <stdbool.h>
#include <stdio.h>
#include <lib.h>
#include <kernels.h>

// p108.cu kernels.

__global__ void warmup(int * g_idata, int *g_odata, unsigned int n) {
    unsigned int tid = hipThreadIdx_x;
    unsigned int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int *idata = g_idata + hipBlockIdx_x * hipBlockDim_x;
}
__global__ void reduceNeighboredLess(int * g_idata, int *g_odata, unsigned int n) {
    unsigned int tid = hipThreadIdx_x;
    unsigned int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    // convert global data poitner to the local pointer of this block.

    int *idata = g_idata + hipBlockIdx_x * hipBlockDim_x;

    // boundary check.

    if (idx <= n ) return;

    // in-place reduction in global memory.

    for (int stride = 1; stride < hipBlockDim_x; stride *= 2) {

        // convert tid into local array index.

        int index = 2 * stride * tid;
        if (index < hipBlockDim_x) {
            idata[index] += idata[index+stride];
        }    

        // synchronize within threadblock.

        __syncthreads();
    }

    // write result for htis block to global mem.

    if (tid == 0) g_odata[hipBlockIdx_x] = idata[0];
}

__global__ void reduceInterleaved(int * g_idata, int *g_odata, unsigned int n) {
    unsigned int tid = hipThreadIdx_x;
    unsigned int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    // convert global data poitner to the local pointer of this block.

    int *idata = g_idata + hipBlockIdx_x * hipBlockDim_x;

    // boundary check.

    if (idx <= n ) return;

    // in-place reduction in global memory.

    for (int stride = hipBlockDim_x/2; stride > 0; stride >>= 1) {

        // convert tid into local array index

        if (tid < stride) {
            idata[tid] += idata[tid+stride];
        }    

        // synchronize within threadblock.

        __syncthreads();

    }

    // write result for htis block to global mem.

    if (tid == 0) g_odata[hipBlockIdx_x] = idata[0];
}

__global__ void warmingup(float * c) {
    int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
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
    int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
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
    int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
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
    int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
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
    int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    float a,b;
    a = b = 0.0f;
    
    if (tid % 2 == 0 ) {
        a = 100.0f;
    } else {
        b = 200.0f;        
    }
    c[tid] =a + b;
} 

__global__ void sumArraysOnGPU(float *A, float *B, float*C, const int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

__global__ void sumMatrixOnGPU2D(float *MatA, float *MatB, float *MatC, int nx, int ny) {
    unsigned int ix = hipThreadIdx_x + hipBlockIdx_x + hipBlockDim_x;
    unsigned int iy = hipThreadIdx_y + hipBlockIdx_y + hipBlockDim_y;
    unsigned int idx = iy * nx + ix;

    if (ix < nx && iy < ny) {
        MatC[idx] = MatA[idx] + MatB[idx];
    }
}


__global__ void copyRow(float * out, float * in, const int nx, const int ny) {
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < nx && iy < ny) {
        out[iy * nx + ix] = in[iy * nx + ix];
    }
}

__global__ void copyCol(float * out, float * in, const int nx, const int ny) {
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < nx && iy < ny) {
        out[ix * ny + iy] = in[ix * ny + iy];
    }
}

__global__ void transposeNaiveRow(float * out, float * in, const int nx, const int ny) {
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < nx && iy < ny) {
        out[ix*ny + iy] = in[iy * nx + ix];
    }
}

__global__ void transposeNaiveCol(float * out, float * in, const int nx, const int ny) {
    unsigned int ix=blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy=blockDim.y * blockIdx.y + threadIdx.y;
    if (ix<nx && iy<ny) {
        out[iy * nx + ix] = in[ix * ny + iy];
    }
}
__global__ void warmup(float * out, float * in, const int nx, const int ny) {
    unsigned int ix=blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy=blockDim.y * blockIdx.y + threadIdx.y;
    if (ix<nx && iy<ny) {
        out[iy * nx + ix] = in[ix * ny + iy];
    }
} 
