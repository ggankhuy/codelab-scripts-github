/*Basic vector sum using cudaMalloc */ 

#include <stdio.h>
#include <cstdlib>

#define N 64
#define ARRSIZE 3
#define LOOPSTRIDE 8
#define STREAMS 4
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

    cudaSetDevice(0);    
    cudaStream_t streams[STREAMS];
    for (int i = 0; i < STREAMS; i ++) {
        int ret = cudaStreamCreate(&streams[i]);
        if (ret != 0) {
            printf("Failure creating stream %u.\n", i);
            return 1;
        } else {
            printf("Stream %u create ok.\n", i);
        }

    }

    a = (int*)malloc(N * sizeof(int));
 	cudaMalloc(&dev_a, N * sizeof(int) );

	for (int i = 0; i < N ; i ++ ) {
		a[i]  = i;
	}

   	//cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    
    const unsigned blocks = 256;
    const unsigned threadsPerBlock = 1;

    k1<<<256,1,0,streams[0]>>>();
    k2<<<256,1,0,streams[1]>>>();
    k3<<<256,1,0,streams[2]>>>();
    cudaDeviceSynchronize();
    k4<<<256,1,0,streams[3]>>>();
//  k3<<<256,1,0,streams[2]>>>();

    //cudaMemcpy(a, dev_a, N * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(dev_a);
    free(a);
    
	return 0;
}
