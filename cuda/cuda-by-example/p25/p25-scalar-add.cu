/**
Simple scalar addition in kernel and returning the result.
 */

// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
//#include <helper_functions.h>
//#include <helper_cuda.h>

__global__ void add( int a, int b, int * c) {
	*c  = a + b;
}

int main ( void ) {
	int c = 200;
	int * dev_c;
    int status = 1000;
	cudaMalloc( ( void ** ) & dev_c, sizeof(int) );
	add <<<1,1>>> (20,70, dev_c);
	status = cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);

    printf("cudaMemcpy status: %d\n.", status); 
	printf( " 20 + 70 = %d\n", c);
}
