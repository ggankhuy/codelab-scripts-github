#include <cstdlib>
#include <cstdio>
#include <stdio.h>

__global__ void fast(int * d_a) {
    size_t start = clock64();
    size_t elapsed = 0;

    while (elapsed < 100000000) {
        elapsed = clock64() - start;
    }
}

__global__ void slow(int *d_a) {
    size_t start = clock64();
    size_t elapsed = 0;

    while (elapsed < 1000000000) {
        elapsed = clock64() - start;
    }
}

int main (void) {
    const int num_iter = 2;
    const int num_streams = 2;
    int *h_a[num_streams];
    int *d_a[num_streams];

    cudaStream_t streams[num_streams];
    size_t numbytes_a = 1000000;

    for (int i = 0 ; i < num_streams ; i++ ) {
        cudaStreamCreate(&streams[i]);
        cudaHostAlloc((void**)&h_a[i], numbytes_a, cudaHostAllocDefault);
        cudaMalloc(&d_a[i], numbytes_a);
    }

    // Expected bahavior if fast_first = 1

    bool fast_first = 0;
    for (int iter = 0 ; iter < num_streams * num_iter ; ++iter) {
        int i = iter % num_streams;
        cudaMemcpyAsync(d_a[i], h_a[i], numbytes_a, cudaMemcpyHostToDevice, streams[i]);
        if (i==fast_first) 
            slow<<<1, 256, 0, streams[i]>>>(d_a[i]);
        else
            fast<<<1, 256, 0, streams[i]>>>(d_a[i]);

        cudaMemcpyAsync(h_a[i], d_a[i], numbytes_a, cudaMemcpyDeviceToHost, streams[i]);
    }

    cudaDeviceSynchronize();

    for (int i =0 ; i < num_streams ; i++ ) {
        cudaStreamDestroy(streams[i]);
        cudaFree(d_a[i]);
    }   

	return 0;
}
