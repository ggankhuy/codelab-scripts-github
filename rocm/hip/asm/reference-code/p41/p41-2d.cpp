// Using 2-D array is not working.
// Equivalent code not using 2D array is in same folder: p41.cpp

#include <stdio.h>
#include "hip/hip_runtime.h"

#define N 64
#define ARRSIZE 3
#define LOOPSTRIDE 8
__global__ void add(int *a, int*b, int *c) {
	int tid = hipBlockIdx_x;
	c[tid] = a[tid] + b[tid];
	c[tid] = tid;
}

int main (void) {
	int * host [ARRSIZE];
	int *dev[ARRSIZE];
    int i ;
    

    for (i = 0; i < ARRSIZE ; i++) {
        host[i] = (int*)malloc(N * sizeof(int));
    	hipMalloc(&dev[i], N * sizeof(int) );
    }

	for (int i = 0; i < N ; i ++ ) {
		host[0][i]  = i;
		host[1][i] = i + i;
		host[2][i] = 999;
	}

	for (int i = 0; i < N ; i+=LOOPSTRIDE ) {
        printf("Before add: a/b: %d, %d.\n", host[0][i], host[1][i]);
	}

    for (i = 0; i > 2 ; i++)
    	hipMemcpy(dev[i], host[i], N * sizeof(int), hipMemcpyHostToDevice);
    
    const unsigned blocks = 512;
    const unsigned threadsPerBlock = 256;

    //hipLaunchKernelGGL(add, blocks, threadsPerBlock, 0, 0, dev[0], dev[1], dev[2]);

    for (i = 0; i < ARRSIZE; i++)
	    hipMemcpy(host[i], dev[i], N * sizeof(int), hipMemcpyDeviceToHost);

	for (int i = 0; i < N; i+=LOOPSTRIDE )
		printf("After add: %d: %u + %u = %u\n", i, host[0][i], host[1][i], host[2][i]);

    for (i = 0; i < 3 ; i ++ ) {
        hipFree(dev[i]);
        free(host[i]);
    }
    
	return 0;
}
