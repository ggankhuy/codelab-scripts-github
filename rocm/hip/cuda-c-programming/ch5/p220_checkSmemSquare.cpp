//#include "../common/common.h"
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <lib.h>
#include <kernels.h>

/*
 * An example of using shared memory to transpose square thread coordinates
 * of a hip grid into a global memory array. Different kernels below
 * demonstrate performing reads and writes with different ordering, as well as
 * optimizing using memory padding.
 */

#ifndef P220
#define P220
#define BDIMX 32
#define BDIMY 32
#define IPAD  1
#endif

int main(int argc, char **argv)
{
    // set up device
    int dev = 0;
    hipDeviceProp_t deviceProp;
    CHECK(hipGetDeviceProperties(&deviceProp, dev));
    printf("%s at ", argv[0]);
    printf("device %d: %s ", dev, deviceProp.name);
    CHECK(hipSetDevice(dev));

    hipSharedMemConfig pConfig;
    CHECK(hipDeviceGetSharedMemConfig ( &pConfig ));
    printf("with Bank Mode:%s ", pConfig == 1 ? "4-Byte" : "8-Byte");

    // set up array size 2048
    int nx = BDIMX;
    int ny = BDIMY;

    bool iprintf = 0;

    if (argc > 1) iprintf = atoi(argv[1]);

    size_t nBytes = nx * ny * sizeof(int);

    // execution configuration
    dim3 block (BDIMX, BDIMY);
    dim3 grid  (1, 1);
    printf("<<< grid (%d,%d) block (%d,%d)>>>\n", grid.x, grid.y, block.x,
           block.y);

    // allocate device memory
    int *d_C;
    CHECK(hipMalloc((int**)&d_C, nBytes));
    int *gpuRef  = (int *)malloc(nBytes);

    CHECK(hipMemset(d_C, 0, nBytes));
    setColReadCol<<<grid, block>>>(d_C);
    CHECK(hipMemcpy(gpuRef, d_C, nBytes, hipMemcpyDeviceToHost));

    if(iprintf)  printData("set col read col   ", gpuRef, nx * ny);

    CHECK(hipMemset(d_C, 0, nBytes));
    setRowReadRow<<<grid, block>>>(d_C);
    CHECK(hipMemcpy(gpuRef, d_C, nBytes, hipMemcpyDeviceToHost));

    if(iprintf)  printData("set row read row   ", gpuRef, nx * ny);

    CHECK(hipMemset(d_C, 0, nBytes));
    setRowReadCol<<<grid, block>>>(d_C);
    CHECK(hipMemcpy(gpuRef, d_C, nBytes, hipMemcpyDeviceToHost));

    if(iprintf)  printData("set row read col   ", gpuRef, nx * ny);

    CHECK(hipMemset(d_C, 0, nBytes));
    setRowReadColDyn<<<grid, block, BDIMX*BDIMY*sizeof(int)>>>(d_C);
    CHECK(hipMemcpy(gpuRef, d_C, nBytes, hipMemcpyDeviceToHost));

    if(iprintf)  printData("set row read col dyn", gpuRef, nx * ny);

    CHECK(hipMemset(d_C, 0, nBytes));
    setRowReadColPad<<<grid, block>>>(d_C);
    CHECK(hipMemcpy(gpuRef, d_C, nBytes, hipMemcpyDeviceToHost));

    if(iprintf)  printData("set row read col pad", gpuRef, nx * ny);

    CHECK(hipMemset(d_C, 0, nBytes));
    setRowReadColDynPad<<<grid, block, (BDIMX + IPAD)*BDIMY*sizeof(int)>>>(d_C);
    CHECK(hipMemcpy(gpuRef, d_C, nBytes, hipMemcpyDeviceToHost));

    if(iprintf)  printData("set row read col DP ", gpuRef, nx * ny);

    // free host and device memory
    CHECK(hipFree(d_C));
    free(gpuRef);

    // reset device
    CHECK(hipDeviceReset());
    return EXIT_SUCCESS;
}
