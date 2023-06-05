/*Basic vector sum using hipMalloc */ 

#include <stdio.h>
#include <cstdlib>
#include "hip/hip_runtime.h"

#define N 64
#define ARRSIZE 3
#define LOOPSTRIDE 8
#define STREAMS 4
__global__ void k1() {
    size_t start = clock64();
    size_t elapsed = 0;
    while (elapsed < 100000000) {
        elapsed = clock64() - start;
    }
}

__global__ void k2() {
    size_t start = clock64();
    size_t elapsed = 0;
    while (elapsed < 300000000) {
        elapsed = clock64() - start;
    }
}
__global__ void k3() {
    size_t start = clock64();
    size_t elapsed = 0;
    while (elapsed < 100000000) {
        elapsed = clock64() - start;
    }
}
__global__ void k4() {
    size_t start = clock64();
    size_t elapsed = 0;
    while (elapsed < 100000000) {
        elapsed = clock64() - start;
    }
}

int main (void) {
    int *a, *b, *c;
    int *dev_a, *dev_b, *dev_c;
    int i ;

    hipSetDevice(0);    
    hipStream_t streams[STREAMS];
    for (int i = 0; i < STREAMS; i ++) {
        int ret = hipStreamCreate(&streams[i]);
        if (ret != 0) {
            printf("Failure creating stream %u.\n", i);
            return 1;
        } else {
            printf("Stream %u create ok.\n", i);
        }

    }

    a = (int*)malloc(N * sizeof(int));
 	hipMalloc(&dev_a, N * sizeof(int) );

	for (int i = 0; i < N ; i ++ ) {
		a[i]  = i;
	}

   	//hipMemcpy(dev_a, a, N * sizeof(int), hipMemcpyHostToDevice);
    
    const unsigned blocks = 256;
    const unsigned threadsPerBlock = 1;

    k1<<<256,1,0,streams[0]>>>();
    hipDeviceSynchronize();
    k2<<<256,1,0,streams[1]>>>();
//  k3<<<256,1,0,streams[2]>>>();

    //hipMemcpy(a, dev_a, N * sizeof(int), hipMemcpyDeviceToHost);

    hipFree(dev_a);
    free(a);
    
	return 0;
}
