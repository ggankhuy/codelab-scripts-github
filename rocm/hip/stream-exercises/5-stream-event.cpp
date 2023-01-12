/*Basic vector sum using hipMalloc */ 

#include <stdio.h>
#include <cstdlib>
#include "hip/hip_runtime.h"

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

int main (void) {
    int *a, *b, *c;
    int *dev_a, *dev_b, *dev_c;
    int i ;

    hipStream_t s1, s2;
    hipStreamCreate(&s1);
    hipStreamCreate(&s2);
    
    // create event. s1 records event. 

    printf("Creating event...\n");
    hipEvent_t e1;
    hipEventCreate(&e1); 
    hipEventRecord(e1, s1);

    const unsigned blocks = 256;
    const unsigned threadsPerBlock = 1;

    printf("Launch streams...\n");
    k1<<<256,1,0,s1>>>();
    hipEventSynchronize(e1);
    k2<<<256,1,0,s2>>>();

	return 0;
}
