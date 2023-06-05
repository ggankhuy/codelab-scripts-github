#include <iostream>
#include <sys/time.h>
#include <lib.h>
#include <kernels.h>

/*
void usage() {
    printf("Usage: ");
    printf("<execname> <p1> <p2> <p3> <p4> where: \n");
    printf("p1: kernel name: \n");
    printf(" - 0:copyrow\n - 1:copyCol\n - 2: transposeRow\n - 3: transposeCol.\n");
    printf("p2 p3: blockx, blocky.\n");
    printf("p4 p5: nx, ny.\n");
    return;
}*/

int main(int argc, char **argv) {
    // setup a device.

    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("%s starting reduction at ", argv[0]);
    printf("device %d: %s", dev, deviceProp.name);
    cudaSetDevice(dev);

    int nElem = 1<<20; // total number of elements to reduce.
    printf(" with array size %d\n", nElem);
   size_t nBytes = nElem * sizeof(float);

    int blocksize = 512;
    int offset = 0;

    if (argc > 1) offset  = atoi(argv[1]);
    if (argc > 2) blocksize = atoi(argv[2]);

    // execution configuration.

    dim3 block(blockx, blocky);
    dim3 grid((nElemm+block.x-1)/block.x, 1);

    // allocate host memory.

    float *h_A = (float*)malloc(nBytes);
    float *h_B = (float*)malloc(nBytes);
    float *hostRef = (float*)malloc(nBytes);
    float *gpuRef = (float*)malloc(nBytes);

    // init host array.

    initialData(h_A, nElem);
    memcpy(h_B, h_A, nBytes);

    // summary at host side

    // sumArraysOnHost(h_A, h_B, hostRef, nElem, offset);
    
    // allocate device memory.

    float *d_A, *d_B, *d_C;
    cudaMalloc((float**)&d_A, nBytes);
    cudaMalloc((float**)&d_B, nBytes);
    cudaMalloc((float**)&d_C, nBytes);

    // copy data from host to device

    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice); //d_B, h_A on book???

    // warmup to avoid start overhead.

    double iStart = seconds();
    warmup <<< grid, block >>> (d_C, d_A, d_C, nElem, offset);
    cudaDeviceSynchronize();
    double iElaps = seconds() - iStart;
    printf("warmup <<< %4d, %4d >>> offset %4d elapsed %f sec.\n", grid.x, block.x, offset, iElaps);
    
    cudaMemcpy*gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
    //checkResult(hostRef, gpuRef, nElem-offset);

    // run kernel.

    iStart = seconds();
    kernel <<<grid, block>>>(d_C, d_A, nx, ny);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;

    // calculate eff. bw.

    float ibnd = 2*nx*ny*sizeof(float)/1e9/iElaps;
    printf("%s elapsed %f sec <<< grid (%d,%d) block (%d,%d) >>> effective bw: %f GB\n", kernelName, iElaps, grid.x, grid.y, block.x, block.y, ibnd);
    
    // check kernel results. skipping.

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    cudaDeviceReset();  

    return 0;


}








