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


__global__ void reduceNeighbored(int * g_idata, int *g_odata, unsigned int n) {
    unsigned int tid = hipThreadIdx_x;

    // convert global data poitner to the local pointer of this block.

    int *idata = g_idata + hipBlockIdx_x * hipBlockDim_x;

    // boundary check.

    if (tid >= n ) return;

    // in-place reduction in global memory.

    for (int stride = 1; stride < hipBlockDim_x; stride *= 2) {

        // convert tid into local array index.

        if ((tid % (2 * stride)) == 0) {
            idata[tid] += idata[tid + stride];
        }    

        // synchronize within threadblock.

        __syncthreads();
    }

    // write result for htis block to global mem.

    if (tid == 0) g_odata[hipBlockIdx_x] = idata[0];
}

__global__ void reduceNeighboredLess(int * g_idata, int *g_odata, unsigned int n) {
    unsigned int tid = hipThreadIdx_x;
    unsigned int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    // convert global data poitner to the local pointer of this block.

    int *idata = g_idata + hipBlockIdx_x * hipBlockDim_x;

    // boundary check.

    if (idx >= n ) return;

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

__global__ void reduceNeighboredInterleaved(int * g_idata, int *g_odata, unsigned int n) {
    unsigned int tid = hipThreadIdx_x;
    unsigned int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    // convert global data poitner to the local pointer of this block.

    int *idata = g_idata + hipBlockIdx_x * hipBlockDim_x;

    // boundary check.

    if (idx >= n ) return;

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


// p232

__global__ void reduceGmem(int * g_idata, int *g_odata, unsigned int n) {
    // set thread id.

    unsigned int tid = hipThreadIdx_x;
    int *idata = g_idata + hipBlockIdx_x + hipBlockDim_x;

    //boundary check.

    unsigned int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
   if (idx >= n) return ;

    // in-place reduction in global memory.

    if (hipBlockIdx_x >= 1024 && tid < 512) idata[tid] += idata[tid+512];
    __syncthreads();
    if (hipBlockIdx_x >= 512 && tid < 256) idata[tid] += idata[tid+256];
    __syncthreads();
    if (hipBlockIdx_x >= 256 && tid < 128) idata[tid] += idata[tid+128];
    __syncthreads();
    if (hipBlockIdx_x >= 128 && tid < 64) idata[tid] += idata[tid+64];
    __syncthreads();

    // unrolling warp

    if (tid < 32) {
        volatile int * vsmem = idata;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    } 

    // write result for this block to global mem.
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

__global__ void warmup(float *A, float *B, float *C, const int n, int offset)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int k = i + offset;

    if (k < n) C[i] = A[k] + B[k];
}

__global__ void readOffset(float *A, float *B, float *C, const int n,
                           int offset)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int k = i + offset;

    if (k < n) C[i] = A[k] + B[k];
}

__global__ void setRowReadRow (int *out)
{
    // static shared memory
    __shared__ int tile[BDIMY][BDIMX];

    // mapping from thread index to global memory index
    unsigned int idx = hipThreadIdx_y * hipBlockDim_x + hipThreadIdx_x;

    // shared memory store operation
    tile[hipThreadIdx_y][hipThreadIdx_x] = idx;

    // wait for all threads to complete
    __syncthreads();

    // shared memory load operation
    out[idx] = tile[hipThreadIdx_y][hipThreadIdx_x] ;
}

__global__ void setColReadCol (int *out)
{
    // static shared memory
    __shared__ int tile[BDIMX][BDIMY];

    // mapping from thread index to global memory index
    unsigned int idx = hipThreadIdx_y * hipBlockDim_x + hipThreadIdx_x;

    // shared memory store operation
    tile[hipThreadIdx_x][hipThreadIdx_y] = idx;

    // wait for all threads to complete
    __syncthreads();

    // shared memory load operation
    out[idx] = tile[hipThreadIdx_x][hipThreadIdx_y];
}

__global__ void setRowReadCol(int *out)
{
    // static shared memory
    __shared__ int tile[BDIMY][BDIMX];

    // mapping from thread index to global memory index
    unsigned int idx = hipThreadIdx_y * hipBlockDim_x + hipThreadIdx_x;

    // shared memory store operation
    tile[hipThreadIdx_y][hipThreadIdx_x] = idx;

    // wait for all threads to complete
    __syncthreads();

    // shared memory load operation
    out[idx] = tile[hipThreadIdx_x][hipThreadIdx_y];
}


__global__ void setRowReadColDyn(int *out)
{
    // dynamic shared memory
    extern  __shared__ int tile[];

    // mapping from thread index to global memory index
    unsigned int row_idx = hipThreadIdx_y * hipBlockDim_x + hipThreadIdx_x;
    unsigned int col_idx = hipThreadIdx_x * hipBlockDim_y + hipThreadIdx_y;

    // shared memory store operation
    tile[row_idx] = row_idx;

    // wait for all threads to complete
    __syncthreads();

    // shared memory load operation
    out[row_idx] = tile[col_idx];
}

__global__ void setRowReadColPad(int *out)
{
    // static shared memory
    __shared__ int tile[BDIMY][BDIMX + IPAD];

    // mapping from thread index to global memory offset
    unsigned int idx = hipThreadIdx_y * hipBlockDim_x + hipThreadIdx_x;

    // shared memory store operation
    tile[hipThreadIdx_y][hipThreadIdx_x] = idx;

    // wait for all threads to complete
    __syncthreads();

    // shared memory load operation
    out[idx] = tile[hipThreadIdx_x][hipThreadIdx_y];
}

__global__ void setRowReadColDynPad(int *out)
{
    // dynamic shared memory
    extern  __shared__ int tile[];

    // mapping from thread index to global memory index
    unsigned int row_idx = hipThreadIdx_y * (hipBlockDim_x + IPAD) + hipThreadIdx_x;
    unsigned int col_idx = hipThreadIdx_x * (hipBlockDim_x + IPAD) + hipThreadIdx_y;

    unsigned int g_idx = hipThreadIdx_y * hipBlockDim_x + hipThreadIdx_x;

    // shared memory store operation
    tile[row_idx] = g_idx;

    // wait for all threads to complete
    __syncthreads();

    // shared memory load operation
    out[g_idx] = tile[col_idx];
}

