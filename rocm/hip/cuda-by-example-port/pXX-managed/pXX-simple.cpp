#include <stdio.h>
#include "hip/hip_runtime.h"

#define N 64
#define ARRSIZE 3
#define LOOPSTRIDE 8
__global__ void add(int *a, int*b, int *c) {
	int tid = hipBlockIdx_x;
	c[tid] = a[tid] + b[tid];
}

/*
int managed_memory = 0;
HIPCHECK(hipDeviceGetAttribute(&managed_memory,
 hipDeviceAttributeManagedMemory,p_gpuDevice));
if (!managed_memory ) {
 printf ("info: managed memory access not supported on the device %d\n Skipped\n",
p_gpuDevice);
}
else {
 HIPCHECK(hipSetDevice(p_gpuDevice));
 HIPCHECK(hipMallocManaged(&Hmm, N * sizeof(T)));
. . .
}*/

int main (void) {
    int *a, *b, *c;
    int i ;

    hipError_t  ret;
    /*
    int * devCount;
    hipGetDeviceCount(devCount);
    printf("No. of devices: %u.\n", devCount);
    
    if (*devCount < 1) {
        printf("Unable to find gpu.\n");
        return 1;
    }
    
    int managed_memory = 0;
    hipDeviceGetAttribute(&managed_memory, hipDeviceAttributeManagedMemory, 0);
    if (!managed_memory ) {
         printf ("info: managed memory access not supported on the device %d\n Skipped\n", 0);
    } 
    return 1;

    hipSetDevice(0);
    */
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
