/*Basic vector sum using hipHostMalloc - zero copy allocation */ 

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
    

    hipHostMalloc((void**)&a, N * sizeof(*a), hipHostMallocMapped | hipHostMallocWriteCombined);
    hipHostMalloc((void**)&b, N * sizeof(*b), hipHostMallocMapped | hipHostMallocWriteCombined);
    hipHostMalloc((void**)&c, N * sizeof(*c), hipHostMallocMapped | hipHostMallocWriteCombined);

    hipHostGetDevicePointer((void**)&dev_a, a, 0);
    hipHostGetDevicePointer((void**)&dev_b, b, 0);
    hipHostGetDevicePointer((void**)&dev_c, c, 0);

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

    hipLaunchKernelGGL(add, blocks, threadsPerBlock, 0, 0, dev_a, dev_b, dev_c);
    hipDeviceSynchronize();

	for (int i = 0; i < N; i+=LOOPSTRIDE )
		printf("After add: %d: %u + %u = %u\n", i, a[i], b[i], c[i]);

    hipFree(dev_a);
    hipFree(dev_b);
    hipFree(dev_c);
    hipHostFree(a);
    hipHostFree(b);
    hipHostFree(c);
    
	return 0;
}
