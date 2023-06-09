
#include <stdio.h>
#include <hip/hip_runtime.h>
#include <sys/time.h>
#include <stdbool.h>
#include <lib.h>
#include <kernels.h>

//#include <lib1.h>

// #define DYN_BUILD

int main(int argc, char ** argv) {
    printf("%s Starting...\n", argv[0]);

    // set up device.

    int dev = 0;
    hipDeviceProp_t deviceProp;
    hipGetDeviceProperties(&deviceProp, dev);
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    hipSetDevice(dev);

    int nx = 1<<14;
    int ny = 1<<14;
    
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);
    printf("Matrix size: nx %d ny %d\n", nx, ny);
    
    // malloc host memory.

    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float*)malloc(nBytes);
    h_B = (float*)malloc(nBytes);
    hostRef = (float*)malloc(nBytes);
    gpuRef = (float*)malloc(nBytes);

    // init data on host side.

    double iStart = seconds();
    initialData(h_A, nxy);
    initialData(h_B, nxy);
    double iElaps = seconds() - iStart;

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // add matrix at host side for result checks

    iStart = seconds();
    sumMatrixOnHost(h_A, h_B, hostRef, nx, ny);
    iElaps = seconds() - iStart;

    // malloc device global memory.

    float *d_MatA, *d_MatB, *d_MatC;
    hipMalloc((void**)&d_MatA, nBytes);
    hipMalloc((void**)&d_MatB, nBytes);
    hipMalloc((void**)&d_MatC, nBytes);

    // trabsfer data from host to device.

    hipMemcpy(d_MatA, h_A, nBytes, hipMemcpyHostToDevice);
    hipMemcpy(d_MatB, h_B, nBytes, hipMemcpyHostToDevice);

    // invoke kernel at host side.

    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x -1 )/block.x, (ny+block.y-1)/block.y);

    iStart = seconds();
    sumMatrixOnGPU2D <<< grid, block >>> (d_MatA, d_MatB, d_MatC, nx, ny);
    hipDeviceSynchronize();
    iElaps = seconds() - iStart;
    
    printf("sumMatrixOnGPU2D <<<(%d, %d), (%d, %d) >>> elapsed %f sec\n", grid.x, grid.y, block.x, block.y, iElaps);

    // copy kernel result back to host side.

    hipMemcpy(gpuRef, d_MatC, nBytes, hipMemcpyDeviceToHost);

    // check device results.

    checkResult(hostRef, gpuRef, nxy);
    
    // free device global memory.

    hipFree(d_MatA);
    hipFree(d_MatB);
    hipFree(d_MatC);

    // free host memory.

    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    // reset device

    hipDeviceReset();
    return 0;
}



