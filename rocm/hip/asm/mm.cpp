/*Basic vector sum using hipMalloc */ 

#include <stdio.h>
#include "hip/hip_runtime.h"
#include <vector>
#include <iostream> 
#include <stdlib.h>
#include <time.h>

#define N 16
#define LOOPSTRIDE 1

__global__ void add(int *a, int*b, int *c) {
	//int tid = hipBlockIdx_x;
	//c[tid] = a[tid] + b[tid];

    int ROW = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int COL = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    float tmpSum = 0;

    if (ROW < N && COL < N) {
        for (int i = 0; i < N; i++) {
            tmpSum += a[ROW * N + i] * b[i * N + COL];
        }
    }
    c[ROW * N + COL] = tmpSum;
}

int main (void) {
    int *h_A, *h_B, *h_C;
    int *dev_a, *dev_b, *dev_c;
    int i ;

    // initialize host array. 

    h_A = (int*)malloc(N * sizeof(int));
    h_B = (int*)malloc(N * sizeof(int));
    h_C = (int*)malloc(N * sizeof(int));

    /*vector<float>h_A(SIZE);
    vector<float>h_B(SIZE);
    vector<float>h_C(SIZE);
    */

    for (int i=0; i<N; i++){
        for (int j=0; j<N; j++){
            h_A[i*N+j] = sin(i);
            h_B[i*N+j] = cos(j);
        }
    }

    // initialize device array.

    hipMalloc(&dev_a, N * sizeof(int) );
    hipMalloc(&dev_b, N * sizeof(int) );
    hipMalloc(&dev_c, N * sizeof(int) );

   	hipMemcpy(dev_a, h_A, N * sizeof(int), hipMemcpyHostToDevice);
   	hipMemcpy(dev_b, h_B, N * sizeof(int), hipMemcpyHostToDevice);
   	hipMemcpy(dev_c, h_C, N * sizeof(int), hipMemcpyHostToDevice);
    
    const unsigned blocks = 256;
    const unsigned threadsPerBlock = 1;

    hipLaunchKernelGGL(add, blocks, threadsPerBlock, 0, 0, dev_a, dev_b, dev_c);

    hipMemcpy(h_A, dev_a, N * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(h_B, dev_b, N * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(h_C, dev_c, N * sizeof(int), hipMemcpyDeviceToHost);

	for (int i = 0; i < N; i+=LOOPSTRIDE )
		printf("After add: %d: %u + %u = %u\n", i, h_A[i], h_B[i], h_C[i]);

    hipFree(dev_a);
    hipFree(dev_b);
    hipFree(dev_c);
    free(h_A);
    free(h_B);
    free(h_C);
    
	return 0;
}
