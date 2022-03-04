/*
Simple vector addition. Mem alloc-d using default cudaHostAlloc.
page locked/pinned: never page out to disk. dma copy: pinned buffer -> gpu memory.
pageable (normal): could page out. dma copy: pageable buffer -> pinned buffer -> gpu memory.
*/

#include <stdio.h>

#define N 8192

__global__ void add(int *a, int*b, int *c) {
	int tid = blockIdx.x;
//	if (tid < N) 
	c[tid] = a[tid] + b[tid];
}

int main (void) {
	int *dev_a, *dev_b, *dev_c;
    int *a, *b, *c;

    cudaHostAlloc((void**)&a, N * sizeof(*dev_a), cudaHostAllocDefault);
    cudaHostAlloc((void**)&b, N * sizeof(*dev_a), cudaHostAllocDefault);
    cudaHostAlloc((void**)&c, N * sizeof(*dev_a), cudaHostAllocDefault);

    cudaMalloc((void**)&dev_a, N * sizeof(*dev_a));
    cudaMalloc((void**)&dev_b, N * sizeof(*dev_b));
    cudaMalloc((void**)&dev_c, N * sizeof(*dev_c));

	for (int i = 0; i < N ; i ++ ) {
		a[i]  = i;
		b[i] = i + 200;
        c[i] = 998; 
	}
	cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

	add<<<N,1>>> (dev_a, dev_b, dev_c);

    cudaThreadSynchronize();
	cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i+=1000 ) {
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
