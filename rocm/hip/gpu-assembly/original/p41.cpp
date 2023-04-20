/*Basic vector sum using hipMalloc */ 

#include <stdio.h>
#include "hip/hip_runtime.h"

#define N 8192
#define N 32
#define ARRSIZE 3
#define LOOPSTRIDE 8
__global__ void add(int *a, int*b, int *c) {
	int tid = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
	c[tid] = a[tid] + b[tid];
}

int main (void) {
    int *a, *b, *c;
    int *dev_a, *dev_b, *dev_c;
    int i ;
    

    a = (int*)malloc(N * sizeof(int));
    b = (int*)malloc(N * sizeof(int));
    c = (int*)malloc(N * sizeof(int));
 	hipMalloc(&dev_a, N * sizeof(int) );
 	hipMalloc(&dev_b, N * sizeof(int) );
 	hipMalloc(&dev_c, N * sizeof(int) );

	for (int i = 0; i < N ; i ++ ) {
		a[i]  = i;
		b[i] = i + i;
		c[i] = 999;
	}

	for (int i = 0; i < N ; i+=LOOPSTRIDE) {
        printf("Before add: a/b: %d, %d.\n", a[i], b[i]);
	}

   	hipMemcpy(dev_a, a, N * sizeof(int), hipMemcpyHostToDevice);
   	hipMemcpy(dev_b, b, N * sizeof(int), hipMemcpyHostToDevice);
   	hipMemcpy(dev_c, c, N * sizeof(int), hipMemcpyHostToDevice);
    
    int BLOCKSIZE=32;
    const unsigned threadsPerBlock = BLOCKSIZE;
    const unsigned blocks = N/BLOCKSIZE;

    printf("blocks/threads: %d, %d.\n", blocks, threadsPerBlock);
    add<<<blocks, threadsPerBlock>>>(dev_a, dev_b, dev_c);

    hipMemcpy(a, dev_a, N * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(b, dev_b, N * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(c, dev_c, N * sizeof(int), hipMemcpyDeviceToHost);

	for (int i = 0; i < N; i+=LOOPSTRIDE)
		printf("After add: %d: %u + %u = %u\n", i, a[i], b[i], c[i]);

    hipFree(dev_a);
    hipFree(dev_b);
    hipFree(dev_c);
    free(a);
    free(b);
    free(c);
    
	return 0;
}
