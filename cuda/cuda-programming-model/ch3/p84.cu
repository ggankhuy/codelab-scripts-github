#include <sys/time.h>
//#include <cuda_runtime.h>
#include <stdbool.h>
#include <stdio.h>

double seconds() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

__global__ void warmingup(float * c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a,b;
    a = b = 0.0f;
    
    if (tid % 2 == 0 ) {
        a = 100.0f;
    } else {
        b = 200.0f;        
    }
    c[tid] =a + b;
} 

__global__ void mathKernel1(float * c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a,b;
    a = b = 0.0f;
    
    if (tid % 2 == 0 ) {
        a = 100.0f;
    } else {
        b = 200.0f;        
    }
    c[tid] =a + b;
} 

__global__ void mathKernel2(float * c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a,b;
    a = b = 0.0f;
    
    if ((tid / warpSize ) % 2  == 0 ) {
        a = 100.0f;
    } else {
        b = 200.0f;        
    }
    c[tid] =a + b;
} 

__global__ void mathKernel3(float * c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a,b;
    a = b = 0.0f;
    
    if (tid % 2 == 0 ) {
        a = 100.0f;
    } else {
        b = 200.0f;        
    }
    c[tid] =a + b;
} 
__global__ void mathKernel4(float * c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a,b;
    a = b = 0.0f;
    
    if (tid % 2 == 0 ) {
        a = 100.0f;
    } else {
        b = 200.0f;        
    }
    c[tid] =a + b;
} 

int main (int argc, char **argv) {
    // setup device.

    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("%s using Device %d: %s\n", argv[0], dev, deviceProp.name);
    
    // setup a data size.

    int size = 64;
    int blocksize = 64;
    if (argc > 1) blocksize = atoi(argv[1]);
    if (argc > 2) size = atoi(argv[2]);

    printf("Data size: %d.\n", size);

    // setup execution configuration.

    dim3 block(blocksize, 1);
    dim3 grid((size+block.x-1)/block.x, 1);
    printf("Execution configured (block %d grid %d\n", block.x, grid.x);

    // alloc gpu memory.

    float *d_C;
    size_t nBytes = size * sizeof(float);
    cudaMalloc((float**)&d_C, nBytes);

    // run a warmup kernel to remove overheard.

    size_t iStart, iElaps;
    cudaDeviceSynchronize();
    iStart = seconds();
    warmingup<<<grid, block>>>(d_C);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    printf("Warmup <<<<%4d %4d >>> elapsed %d sec \n", grid.x, block.x, iElaps);

    // run kernel 1

    iStart = seconds();
    mathKernel1<<<grid, block>>>(d_C);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    printf("MathKernel1 <<<%4d %4d >>> elapsed %d sec \n", grid.x, block.x, iElaps);

    // run kernel 2

    iStart = seconds();
    mathKernel2<<<grid, block>>>(d_C);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    printf("MathKernel2 <<<%4d %4d >>> elapsed %d sec \n", grid.x, block.x, iElaps);

    /*
    // run kernel 3

    iStart = seconds();
    mathKernel1<<<grid, block>>>(d_C);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    printf("MathKernel3 <<<%4d %4d >>> elapsed %d sec \n", grid.x, block.x, iElaps);

    // run kernel 4

    iStart = seconds();
    mathKernel1<<<grid, block>>>(d_C);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    printf("MathKernel4 <<<%4d %4d >>> elapsed %d sec \n", grid.x, block.x, iElaps);
    */
}
