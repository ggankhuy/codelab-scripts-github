#include <stdio.h>
#include <hip/hip_runtime.h>
#include <sys/time.h>
#include <stdbool.h>
#include <lib.h>
#include <kernels.h>

int main(int argc, char ** argv) {
    printf("%s Starting...\n", argv[0]);

    // set up device.

    int dev = 0;
    hipDeviceProp_t deviceProp;
    hipGetDeviceProperties(&deviceProp, dev);
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    hipSetDevice(dev);

    bool bresult = false;

    int size = 1<<24;
    printf("array size: size %d\n", size);

    int blocksize = 512;

    if (argc > 1 ) {
        blocksize = atoi(argv[1]);
    }
    

    dim3 block(blocksize, 1);
    dim3 grid ((size+block.x-1)/block.x, 1);

    printf("Grid %d block %d.\n", grid.x, block.x);

    // malloc host memory.

    size_t bytes = size * sizeof(int);

    int *h_idata = (int*)malloc(bytes);
    int *h_odata = (int*)malloc(grid.x*sizeof(int));
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
    hipMalloc((void**)&d_idata, bytes);
    hipMalloc((void**)&d_odata, grid.x*sizeof(int));

    // cpu reduction, skipping....

    iStart = seconds();
    int cpu_sum = recursiveReduce(tmp, size);
    iElaps = seconds() - iStart;
    printf("cpu reduce elapsed %d ms cpu sum: %d.\n", iElaps, cpu_sum);

    // kernel1. recursive Reduce.

    hipMemcpy(d_idata, h_idata, bytes, hipMemcpyHostToDevice);
    hipDeviceSynchronize();
    iStart = seconds();
    warmup<<<grid, block>>>(d_idata, d_odata, size);
    hipDeviceSynchronize();
    iElaps = seconds() - iStart;
    hipMemcpy(h_odata, d_odata, grid.x*sizeof(int), hipMemcpyDeviceToHost);
    int gpu_sum=0;

    for (int i = 0; i < grid.x; i++) { gpu_sum += h_odata[i]; }
    
    printf("gpu warmup elapsed %d ms gpu_sum: %d '<<<grid %d block %d>>>\n", iElaps, gpu_sum, grid.x, block.x);

    // kern1l reduce Neighbored.

    hipMemcpy(d_idata, h_idata, bytes, hipMemcpyHostToDevice);
    hipDeviceSynchronize();
    iStart = seconds();
    reduceGmem<<<grid, block>>>(d_idata, d_odata, size);
    hipDeviceSynchronize();
    iElaps = seconds() - iStart;
    hipMemcpy(h_odata, d_odata, grid.x*sizeof(int), hipMemcpyDeviceToHost);
    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++ ) { gpu_sum += h_odata[i];}
    printf("gpu reduceNeighbored: elapsed %d ms gpu_sum: %d <<<grid %d block %d>>>\n", iElaps, gpu_sum, grid.x, block.x);

    hipFree(d_idata);
    hipFree(d_odata);

    // free host memory.

    free(h_idata);
    free(h_odata);

    // reset device

    hipDeviceReset();
    return 0;
}



