#include <sys/time.h>
#include <hip/hip_runtime.h>
#include <stdbool.h>
#include <stdio.h>
#include <lib.h>
#include <kernels.h>

int main (int argc, char **argv) {
    // setup device.

    int dev = 0;
    hipDeviceProp_t deviceProp;
    hipGetDeviceProperties(&deviceProp, dev);
    printf("%s using Device %d: %s\n", argv[0], dev, deviceProp.name);
    
    // setup a data size.

    int size = 16384 * 1024;
    int blocksize = 1024;
    if (argc > 1) blocksize = atoi(argv[1]);
    if (argc > 2) size = atoi(argv[2]);

    printf("Data size: %d.\n", size);

    // setup execution configuration.

    dim3 block(blocksize, 1);
    dim3 grid((size+block.x-1)/block.x, 1);
    printf("Execution configured (block %d grid %d).\n", block.x, grid.x);

    // alloc gpu memory.

    float *d_C;
    size_t nBytes = size * sizeof(float);
    hipMalloc((float**)&d_C, nBytes);

    // run a warmup kernel to remove overheard.

    size_t iStart, iElaps;
    hipDeviceSynchronize();
    iStart = seconds();
    warmingup<<<grid, block>>>(d_C);
    hipDeviceSynchronize();
    iElaps = seconds() - iStart;
    printf("Warmup <<<<%4d %4d >>> elapsed %d sec \n", grid.x, block.x, iElaps);

    // run kernel 1

    iStart = seconds();
    mathKernel1<<<grid, block>>>(d_C);
    hipDeviceSynchronize();
    iElaps = seconds() - iStart;
    printf("MathKernel1 <<<%4d %4d >>> elapsed %d sec \n", grid.x, block.x, iElaps);

    // run kernel 2

    iStart = seconds();
    mathKernel2<<<grid, block>>>(d_C);
    hipDeviceSynchronize();
    iElaps = seconds() - iStart;
    printf("MathKernel2 <<<%4d %4d >>> elapsed %d sec \n", grid.x, block.x, iElaps);

    
    // run kernel 3

    iStart = seconds();
    mathKernel1<<<grid, block>>>(d_C);
    hipDeviceSynchronize();
    iElaps = seconds() - iStart;
    printf("MathKernel3 <<<%4d %4d >>> elapsed %d sec \n", grid.x, block.x, iElaps);

    // run kernel 4

    iStart = seconds();
    mathKernel1<<<grid, block>>>(d_C);
    hipDeviceSynchronize();
    iElaps = seconds() - iStart;
    printf("MathKernel4 <<<%4d %4d >>> elapsed %d sec \n", grid.x, block.x, iElaps);
    
}
