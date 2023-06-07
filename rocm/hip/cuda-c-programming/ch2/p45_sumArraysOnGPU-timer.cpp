#include <hip/hip_runtime.h>
#include <stdio.h>
#include <sys/time.h>
#include <lib.h>
#include <kernels.h>

/*
#define CHECK(call)
{
    const hipError_t error = call;
    if (error != hipSuccess) {
        printf("Error: %s:%d, ", __FILE__, __LINE);
        printf("code: %d, reason: %s\n", error, hipErrorString(error));
        exit(1);
    }
}
*/

int main(int argc, char **argv) {
    printf("%s Starting...\n", argv[0]);

    // setup device.

    int dev = 0;
    hipDeviceProp_t deviceProp;
    hipGetDeviceProperties(&deviceProp, dev);
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    hipSetDevice(dev);

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
    hipMalloc((float**)&d_A, nBytes);
    hipMalloc((float**)&d_B, nBytes);
    hipMalloc((float**)&d_C, nBytes);

    // transfer data from host to device

    hipMemcpy(d_A, h_A, nBytes, hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B, nBytes, hipMemcpyHostToDevice);

    // invoke kernel at host side.

    int iLen = 1024;
    dim3 block(iLen);
    dim3 grid((nElem + block.x - 1)/block.x);
    
    iStart = seconds();
    sumArraysOnGPU <<<grid, block>>>(d_A, d_B, d_C, nElem);
    hipDeviceSynchronize();
    iElaps = seconds() - iStart;
    printf("sumArraysOnGPU<<<%d, %d>>> Time elapsed %f sec\n", grid.x, block.x, iElaps);

    // copy kernel result back to host side

    hipMemcpy(gpuRef, d_C, nBytes, hipMemcpyDeviceToHost);
    
    // free device global memory

    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);

    // free host memory.

    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    return 0;
}
