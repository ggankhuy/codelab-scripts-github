/* Simple vector addition. 
The block, grid dimension are calc-d automatically.
*/

#include <stdio.h>
#define N 1024
#define MAX_THREAD_PER_BLOCK 1024

__global__ void add( int * a, int * b, int * c ) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < N)
       	c[tid] = a[tid] + b[tid];
}

int main (void) {
	int a[N], b[N], c[N];
	int *dev_a, *dev_b, *dev_c;

	// allocate dev memory for N size for pointers declared earlier.

    printf("\nAllocating memory...");
	cudaMalloc( (void**)&dev_a, N * sizeof(int));
	cudaMalloc( (void**)&dev_b, N * sizeof(int));
	cudaMalloc( (void**)&dev_c, N * sizeof(int));

	for (int i = 0; i < N; i++) {
		a[i] = i;
		b[i] = i+2;
	}

	// copy the initialized local memory values to device memory. 

    printf("\nCopy host to device...");
	cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

	// invoke the kernel: 
	// block count: (N+127)/128
	// thread count: 128
    
    printf("\nLaunch cuda kernel...");
    //add<<<(N+MAX_THREAD_PER_BLOCK-1)/MAX_THREAD_PER_BLOCK, MAX_THREAD_PER_BLOCK>>> (dev_a, dev_b, dev_c);
    add<<<N/MAX_THREAD_PER_BLOCK, MAX_THREAD_PER_BLOCK>>> (dev_a, dev_b, dev_c);

    printf("\nCopy back from GPU to host...");
	cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);

	for (int i = 0; i < N; i+=50) {
		printf("%d + %d = %d\n", a[i], b[i], c[i]);
	}

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
}
