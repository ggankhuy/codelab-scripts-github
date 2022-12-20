/*Basic vector sum using hipMalloc */ 

#include <stdio.h>
#include "hip/hip_runtime.h"

#define N 64
#define ARRSIZE 3
#define LOOPSTRIDE 8

int main (void) {
    int *a, *b, *c;
    int *dev_a, *dev_b, *dev_c;
    int i ;
    

 	hipMalloc(&dev_a, N * sizeof(int) );
    hipFree(dev_a);
	return 0;
}
