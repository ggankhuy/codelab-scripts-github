/*Basic vector sum involving shared memory to hold sums */ 

#include <stdio.h>
#include "hip/hip_runtime.h"

#define imin(a,b) (a<b?a:b)

const int N = 512;
const int threadsPerBlock = 128;
const int blocksPerGrid = imin (32, (N+threadsPerBlock-1)/(threadsPerBlock));
#define LOOPSTRIDE 64
__global__ void dot(int * a, int *b, int *c) {

    // gid = index into array relative to global memory (vram).
    // lid = index into array relative to current block.

    int gid = hipThreadIdx_x  + hipBlockIdx_x * hipBlockDim_x;
    int lid = hipThreadIdx_x;

    // will be created on every block.

    __shared__ int arrShared[threadsPerBlock];

    arrShared[lid] = a[gid] + b[gid];

    // do a lot of computations on arrShared. (not implemented).

    __syncthreads();

    // copy result to vram. 

    c[gid] = arrShared[lid];
}

int main( void ) {

    // partial c holds array len of block size.

    int *a, *b, *c;
    int *dev_a, *dev_b, *dev_c;
 
    a = (int*) malloc(N * sizeof(int));
    b = (int*) malloc(N * sizeof(int));
    c = (int*) malloc(N * sizeof(int));

    hipMalloc((void**)&dev_a, N * sizeof(int));
    hipMalloc((void**)&dev_b, N * sizeof(int));
    hipMalloc((void**)&dev_c, N * sizeof(int));

    for (int i = 0; i < N ; i++) {
        a[i] = i;
        b[i] = i + 2;
        c[i] = 999;
    }

    for (int i = 0; i < N; i+=LOOPSTRIDE )
        printf("Before add: %d: %u + %u = %u\n", i, a[i], b[i], c[i]);

    hipMemcpy(dev_a, a, N*sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(dev_b, b, N*sizeof(int), hipMemcpyHostToDevice);

    dot <<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_c);

    hipMemcpy(c, dev_c, N*sizeof(int), hipMemcpyDeviceToHost);

    for (int i = 0; i < N; i+=LOOPSTRIDE )
        printf("After add: %d: %u + %u = %u\n", i, a[i], b[i], c[i]);

    hipFree(dev_a);
    hipFree(dev_b);
    hipFree(dev_c);
    
    free(a);
    free(b);
    free(c);
}
