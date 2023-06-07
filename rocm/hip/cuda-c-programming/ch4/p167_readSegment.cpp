#include <iostream>
#include <sys/time.h>
#include <lib.h>
#include <kernels.h>
#include <hip/hip_runtime.h>


void usage() {
    printf("Example for mem-misailgned read...");
    printf("Usage: ");
    printf("<execname> <p1> where: \n");
    printf("p1: offset from alignment.\n");
    return;

}

int main(int argc, char **argv) {

    // setup a device.

    int dev = 0;
    hipDeviceProp_t deviceProp;
    hipGetDeviceProperties(&deviceProp, dev);
    printf("%s starting reduction at ", argv[0]);
    printf("device %d: %s", dev, deviceProp.name);
    hipSetDevice(dev);

    // total number of elements to reduce and assoc-d bytes.

    int nElem = 1<<20;                          
    printf(" with array size %d\n", nElem);
    size_t nBytes = nElem * sizeof(float);

    int blocksize = 512;
    int offset = 0;

    if (argc > 1) offset  = atoi(argv[1]);
    if (argc > 2) blocksize = atoi(argv[2]);

    printf("offset: %d, blocksize: %d.\n", offset, blocksize);

    // Execution configuration.

    dim3 block(blocksize, 1);
    dim3 grid((nElem+block.x-1)/block.x, 1);

    // allocate host memory.

    float *h_A = (float*)malloc(nBytes);
    float *h_B = (float*)malloc(nBytes);
    float *hostRef = (float*)malloc(nBytes);
    float *gpuRef = (float*)malloc(nBytes);

    // init host array.

    initialData(h_A, nElem);
    memcpy(h_B, h_A, nBytes);

    // sum array at host side.

    sumArraysOnHost(h_A, h_B, hostRef, nElem, offset);
    
    // allocate device memory.

    float *d_A, *d_B, *d_C;
    hipMalloc((float**)&d_A, nBytes);
    hipMalloc((float**)&d_B, nBytes);
    hipMalloc((float**)&d_C, nBytes);

    // copy data from host to device

    hipMemcpy(d_A, h_A, nBytes, hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_A, nBytes, hipMemcpyHostToDevice); //d_B, h_A on book???

    // warmup to avoid start overhead.

    double iStart = seconds();
    warmup <<< grid, block >>> (d_C, d_A, d_C, nElem, offset);
    hipDeviceSynchronize();
    double iElaps = seconds() - iStart;
    printf("warmup <<< %4d, %4d >>> offset %4d elapsed %f sec.\n", grid.x, block.x, offset, iElaps);
    
    hipMemcpy(gpuRef, d_C, nBytes, hipMemcpyDeviceToHost);

    // run kernel.

    iStart = seconds();
    readOffset<<<grid, block>>>(d_A, d_B, d_C, nElem, offset);
    hipDeviceSynchronize();
    iElaps = seconds() - iStart;

    printf("readOffset <<< %4d, %4d >>> offset %4d elapsed %f sec\n", grid.x, block.x, offset, iElaps);

    hipMemcpy(gpuRef, d_C, nBytes, hipMemcpyDeviceToHost);
    checkResult(hostRef, gpuRef, nElem-offset);

    // check kernel results. skipping.

    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    free(h_A);
    free(h_B);
    hipDeviceReset();  

    return 0;


}








