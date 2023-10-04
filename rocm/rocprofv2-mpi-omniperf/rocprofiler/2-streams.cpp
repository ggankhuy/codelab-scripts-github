// reproducer.cpp

#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>

__global__ void x10(int * d_a) {
    size_t start = clock64();
    size_t elapsed = 0;
    while (elapsed < 100000000) {
        elapsed = clock64() - start;
    }
}

__global__ void x1(int * d_a) {
    size_t start = clock64();
    size_t elapsed = 0;
    while (elapsed < 1000000000) {
        elapsed = clock64() - start;
    }
}

int main(int argc, char **argv) {
    const int num_iter = 2;
    const int num_streams = 2;
    int *h_a[num_streams];
    int *d_a[num_streams];
    size_t numbytes_a = 1000000;
    hipStream_t streams[num_streams];
    for (int i = 0; i < num_streams; i++) {
        hipStreamCreate(&streams[i]);
        hipHostMalloc((void**)&h_a[i], numbytes_a);
        hipMalloc(&d_a[i], numbytes_a);
    }      
    // Expected behavior if x10_first = 1
    bool x10_first = 0;
    for (int iter = 0; iter < num_streams * num_iter; ++iter) {
        int i = iter % num_streams;
        hipMemcpyAsync(d_a[i], h_a[i], numbytes_a, hipMemcpyHostToDevice, streams[i]);
        if (i == x10_first)
            x1<<<1, 256, 0, streams[i]>>>(d_a[i]);
        else
            x10<<<1, 256, 0, streams[i]>>>(d_a[i]);
        hipMemcpyAsync(h_a[i], d_a[i], numbytes_a, hipMemcpyDeviceToHost, streams[i]);
    }

    hipDeviceSynchronize();

    for (int i = 0; i < num_streams; i++) {
        hipStreamDestroy(streams[i]);
        hipFree(d_a[i]);
    }
}
