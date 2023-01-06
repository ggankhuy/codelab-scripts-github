/*
https://web.microsoftstream.com/video/6b3424a4-5197-447e-9e69-96656982b36d?referrer=https:%2F%2Fconfluence.amd.com%2F
~32m:38s*/
/*Basic vector sum using hipMalloc */ 

#include <stdio.h>
#include "hip/hip_runtime.h"

#define N 64
#define ARRSIZE 3
#define LOOPSTRIDE 8
__global__ void add(int *a, int*b, int *c) {
	int tid = hipBlockIdx_x;
	c[tid] = a[tid] + b[tid];
}

int main (void) {
    int *a, *b, *c;
    int *dev_a, *dev_b, *dev_c;
    int i ;
    

    a = (int*)malloc(N * sizeof(int));
 	hipMalloc(&dev_a, N * sizeof(int) );

	for (int i = 0; i < N ; i ++ ) {
		a[i]  = i;
	}

   	hipMemcpy(dev_a, a, N * sizeof(int), hipMemcpyHostToDevice);
    
    const unsigned blocks = 256;
    const unsigned threadsPerBlock = 1;

    //hipLaunchKernelGGL(add, blocks, threadsPerBlock, 0, 0, dev_a, dev_b, dev_c);

    hipMemcpy(a, dev_a, N * sizeof(int), hipMemcpyDeviceToHost);

    hipFree(dev_a);
    free(a);
    
	return 0;
}
