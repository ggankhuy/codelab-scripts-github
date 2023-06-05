#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>
#include <lib.h>
#include <kernels.h>

/*
#define CHECK(call)
{
    const cudaError_t error = call;
    if (error != cudaSuccess) {
        printf("Error: %s:%d, ", __FILE__, __LINE);
        printf("code: %d, reason: %s\n", error, cudaErrorString(error));
        exit(1);
    }
}
*/

void sumArraysOnHost(float * A, float *B, float *C, const int N) {
    for (int idx = 0; idx < N ; idx++) 
        C[idx] = A[idx] + B[idx];
}

__global__ void sumArraysOnGPU(float *A, float *B, float*C, const int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

int main(int argc, char **argv) {
    printf("%s Starting...\n", argv[0]);

    // setup device.

    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    cudaSetDevice(dev);

    // setup device size of vectors.

    int nElem = 1 << 24 ;
    printf("Vector size %d\n", nElem);
    
    // malloc host memory.
    
    size_t nBytes = nElem * sizeof(float);

    float *h_A, *h_B,*hostRef, *gpuRef;
    h_A = (float*)malloc(nBytes);
    h_B = (float*)malloc(nBytes);
    hostRef = (float*)malloc(nBytes);
    gpuRef = (float*)malloc(nBytes);

    double iStart, iElaps;

    // initialize data at host side.

    iStart = seconds();
    initialData(h_A, nElem);
    initialData(h_B, nElem);
    iElaps = seconds() - iStart;

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // add vector at host side for result checks.

    iStart = seconds();
    sumArraysOnHost(h_A, h_B, hostRef, nElem);
    iElaps = seconds()  - iStart;
    
    // malloc device global memory.

    float *d_A, *d_B, *d_C;
    cudaMalloc((float**)&d_A, nBytes);
    cudaMalloc((float**)&d_B, nBytes);
    cudaMalloc((float**)&d_C, nBytes);

    // transfer data from host to device

    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

    // invoke kernel at host side.

    int iLen = 1024;
    dim3 block(iLen);
    dim3 grid((nElem + block.x - 1)/block.x);
    
    iStart = seconds();
    sumArraysOnGPU <<<grid, block>>>(d_A, d_B, d_C, nElem);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    printf("sumArraysOnGPU<<<%d, %d>>> Time elapsed %f sec\n", grid.x, block.x, iElaps);

    // copy kernel result back to host side

    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
    
    // free device global memory

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // free host memory.

    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    return 0;
}
