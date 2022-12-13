/*
Simple vector addition. Mem alloc-d using default cudaMalloc.
*/
#include <stdio.h>
#include <cstlib.h>
#include <cstdio.h>

#define N 8192

__global__ void add(int *a, int*b, int *c) {
	int tid = blockIdx.x;
//	if (tid < N) 
	c[tid] = a[tid] + b[tid];
}

int main (void) {
	//int a[N], b[N], c[N];
	int *dev_a, *dev_b, *dev_c;
    int *a, *b, *c;

	cudaMalloc( (void**)&dev_a, N * sizeof(int) );
	cudaMalloc( (void**)&dev_b, N * sizeof(int) );
	cudaMalloc( (void**)&dev_c, N * sizeof(int) );

    a = (int*)malloc( N * sizeof(float));
    b = (int*)malloc( N * sizeof(float));
    c = (int*)malloc( N * sizeof(float));

	for (int i = 0; i < N ; i ++ ) {
		a[i]  = i;
		b[i] = i + 200;
	}
	cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

	add<<<N,1>>> (dev_a, dev_b, dev_c);

	cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);

	for (int i = 0; i < N; i+=1000 ) {
		printf("%d: %d + %d = %d\n", i, a[i], b[i], c[i]);
	}

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	return 0;
}
