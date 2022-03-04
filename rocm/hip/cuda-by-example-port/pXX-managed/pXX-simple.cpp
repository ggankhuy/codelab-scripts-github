/*Basic vector sum using managedMemory hipMallocManaged */ 

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
    int i ;

    hipError_t  ret;

    ret = hipMallocManaged(&a, N * sizeof(*a));
    ret = hipMallocManaged(&b, N * sizeof(*b));
    ret = hipMallocManaged(&c, N * sizeof(*c));

	for (int i = 0; i < N ; i ++ ) {
		a[i]  = i;
		b[i] = i + i;
		c[i] = 999;
	}

	for (int i = 0; i < N ; i+=LOOPSTRIDE ) {
        printf("Before add: a/b: %d, %d.\n", a[i], b[i]);
  	}

    const unsigned blocks = 256;
    const unsigned threadsPerBlock = 1;

    hipLaunchKernelGGL(add, blocks, threadsPerBlock, 0, 0, a, b, c);
    hipDeviceSynchronize();

	for (int i = 0; i < N; i+=LOOPSTRIDE )
		printf("After add: %d: %u + %u = %u\n", i, a[i], b[i], c[i]);

    
    hipFreeHost(a);
    hipFreeHost(b);
    hipFreeHost(c);
    
	return 0;
}
