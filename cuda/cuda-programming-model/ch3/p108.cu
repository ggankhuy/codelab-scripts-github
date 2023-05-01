
#include <stdio.h>
#include <cuda_runtime.h>
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
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    cudaSetDevice(dev);

    bool bresult = false;

    int size = 1<<24;
    printf("array size: nx %d ny %d\n", nx, ny);

    int blocksize = 512;

    if (argv > 1 ) {
        blocksize = atoi(argv[1]);
    }
    

    dim3 block(blocksize, 1);
    dim3 grid ((size+block.x-1)/block.x, 1)

    printf("Grid %d block %d.\n", grid.x, block.x);

    // malloc host memory.

    size_t bytes = size * sizeof(int);

    int h_idata = (int*)malloc(bytes);
    int h_odata = (int*)malloc(grid.x*sizeof(int));
    int * tmp  = (int*)malloc(bytes);

    // init data on host side.

    for (int i = 0; i < size; i++) {
        h_idata[i]=(int)(rand() & 0xff);
    }

    memcpy(tmp, h_idata, bytes);

    size_t iStart, iElaps;
    int gpu_num = 0;

    // allocate device memory.

    int *d_idata, *d_odata;
    cudaMalloc((void**)&d_idata, bytes);
    cudaMalloc((void**)&d_odata, bytes);

    // cpu reduction, skipping....

    // kernel1. recursive Reduce.

    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    iStart = seconds();

    warmup<<<grid, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();

    iElapse = seconds() - iStart;

    cudaMemcpy(h_odata, d_odata, grid.x*sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum=0;

    for (int i = 0;i < grid.x, i++) { gpu_sum += h_odata[i]; }
    
    print("gpu warmup elapsed %d ms gpu_sum: $d <<<grid %d block %d>>>\n", iElaps, gpu_sum, grid.x, block.x);

    // kern1l reduce Neighbored.
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = seconds();
    reduceNeighbored<<<grid, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iStart = seconds();
    cudaMemcpyCopy(h_odata, d_odata, grid.x*sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++ ) { gpu_sum += h_odata[i];}
    printf("gpu neighbored: elapsed %d ms gpu_sum: %d <<<grid %d block %d>>>\n", iElaps, gpu_sum, grid.x, block.x);

    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;

    cudaMemcpy(h_odata, d_odata, grid.x/8*sizeof(int), cudamemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x/8; i++) { gpu_sum += h_odata[i];}
    printf("gpu Cmptnroll elapsed %d ms gpu_sum: $d <<<grid %d block %d>>>\n", iElaps, gpu_sum, grid.x, block.x);

    cudaFree(d_idata);
    cudaFree(d_odata);

    // free host memory.

    free(h_idata);
    free(h_odata);

    // reset device

    cudaDeviceReset();
    return 0;
}



