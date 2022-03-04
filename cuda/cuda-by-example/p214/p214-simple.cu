/*
Simple vector addition. Mem alloc-d using default cudaMalloc.
*/
#include <stdio.h>

#define N 8192

__global__ void add(int *a, int*b, int *c) {
    //int tid = threadIdx.x * blockIdx.x * blockDim.x;
    int tid = blockIdx.x;
	if (tid < N) 
	    c[tid] = a[tid] + b[tid];
}

int main (void) {
	int *dev_a, *dev_b, *dev_c;
    int *a, *b, *c;

    cudaHostAlloc((void**)&a, N * sizeof(int), cudaHostAllocWriteCombined|cudaHostAllocMapped);
    cudaHostAlloc((void**)&b, N * sizeof(int), cudaHostAllocWriteCombined|cudaHostAllocMapped);
    cudaHostAlloc((void**)&c, N * sizeof(int), cudaHostAllocWriteCombined|cudaHostAllocMapped);

    cudaHostGetDevicePointer((void**)&dev_a, a, 0);
    cudaHostGetDevicePointer((void**)&dev_b, b, 0);
    cudaHostGetDevicePointer((void**)&dev_c, c, 0);

	for (int i = 0; i < N ; i ++ ) {
		a[i]  = i;
		b[i] = i + 200;
        c[i] = 999;
	}

	add<<<N,1>>> (dev_a, dev_b, dev_c);

    cudaThreadSynchronize();
    //cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);

	for (int i = 0; i < N; i+=1000 ) {
		printf("%d: %d + %d = %d\n", i, dev_a[i], dev_b[i], dev_c[i]);
		printf("%d: %d + %d = %d\n", i, a[i], b[i], c[i]);
	}

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
    cudaFreeHost(a);
    cudaFreeHost(b);
    cudaFreeHost(c);
	return 0;
}
