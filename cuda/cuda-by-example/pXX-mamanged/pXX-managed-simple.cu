/*
Simple vector addition. Mem alloc-d using default cudaMallocManaged / unified addressing.
Check unified memory vs. unified virtual addressing 
https://developer.nvidia.com/blog/unified-memory-in-cuda-6/
*/

#include <stdio.h>

#define N 8192

__global__ void add(int *a, int*b, int *c) {
	int tid = blockIdx.x;
	if (tid < N) 
    	c[tid] = a[tid] + b[tid];
}

int main (void) {
	int *dev_a, *dev_b, *dev_c;
	cudaMallocManaged( (void**)&dev_a, N * sizeof(int) );
	cudaMallocManaged( (void**)&dev_b, N * sizeof(int) );
	cudaMallocManaged( (void**)&dev_c, N * sizeof(int) );

	for (int i = 0; i < N ; i ++ ) {
		dev_a[i]  = i;
		dev_b[i] = i + 200;
		dev_c[i] = 997;
	}
	add<<<N,1>>> (dev_a, dev_b, dev_c);
    cudaDeviceSynchronize();
    //cudaThreadSynchronize(); // Decrecated by cudaDeviceSynchronize().

	for (int i = 0; i < N; i+=1000 ) {
		printf("%d: %d + %d = %d\n", i, dev_a[i], dev_b[i], dev_c[i]);
	}

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	return 0;
}
