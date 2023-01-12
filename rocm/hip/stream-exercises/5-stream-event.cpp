/*Basic vector sum using hipMalloc */ 

#include <stdio.h>
#include <cstdlib>
#include "hip/hip_runtime.h"

#define N 64
#define ARRSIZE 3
#define LOOPSTRIDE 8
#define STREAMS 2
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

    // create streams. 
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

    // create event. s1 records event. 

    hipEvent_t e1;
    hipEventCreate(&e1); 
    hipEventRecord(e1, streams[0]);

    const unsigned blocks = 256;
    const unsigned threadsPerBlock = 1;

    k1<<<256,1,0,streams[0]>>>();
    hipEventSynchronize(e1);
    k2<<<256,1,0,streams[1]>>>();

	return 0;
}
