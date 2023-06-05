// reproducer-2.cpp
// delta from original reproduce.cpp is attempt to ignore default stream 0.

#include <cstdio>
#include <cstdlib>
//#include <[hip/hip_runtime.h>
//#define N 1000000
#define N 4

__global__ void fast( int * a, int * b, int * c) {
    size_t start = clock64();
    size_t elapsed = 0;
    while (elapsed < 100000000) {
        elapsed = clock64() - start;
    }
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N)
       	c[tid] = a[tid] + b[tid];
        c[tid] = 1005;
}

__global__ void fast2x( int * a, int * b, int * c) {
    size_t start = clock64();
    size_t elapsed = 0;
    while (elapsed < 200000000) {
        elapsed = clock64() - start;
    }
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N)
       	c[tid] = a[tid] + b[tid];
        c[tid] = 1005;
}

__global__ void slow( int * a, int * b, int * c) {
    size_t start = clock64();
    size_t elapsed = 0;
    while (elapsed < 1000000000) {
        elapsed = clock64() - start;
    }
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N)
       	c[tid] = a[tid] + b[tid];
        c[tid] = 1005;
}

__global__ void slow2x( int * a, int * b, int * c) {
    size_t start = clock64();
    size_t elapsed = 0;
    while (elapsed < 2000000000) {
        elapsed = clock64() - start;
    }
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N)
       	c[tid] = a[tid] + b[tid];
        c[tid] = 1005;
}

int main(int argc, char **argv) {
    const int num_iter = 2;
    const int num_streams = 2;

    int *h_a[num_streams];
    int *d_a[num_streams];

    int *h_b[num_streams];
    int *d_b[num_streams];

    int *h_c[num_streams];
    int *d_c[num_streams];

    cudaStream_t streams[num_streams];

    // Create streams and allocate host and device memory for each.

    for (int s = 0; s < num_streams; s++) {
        //cudaStreamCreate(&streams[i]);
        cudaStreamCreateWithFlags(&streams[s], cudaStreamNonBlocking);

        cudaHostAlloc((void**)&h_a[s], N, cudaHostAllocDefault);
        cudaHostAlloc((void**)&h_b[s], N, cudaHostAllocDefault);
        cudaHostAlloc((void**)&h_c[s], N, cudaHostAllocDefault);

        cudaMalloc(&d_a[s], N);
        cudaMalloc(&d_b[s], N);
        cudaMalloc(&d_c[s], N);

        for (int j = 0; j < N; j++) {
            h_a[s][j]  = j + num_streams * 1000 * s;
            h_b[s][j]  = j + num_streams * 2000 * s;
        }
    }      

    for (int s = 0; s < num_streams; s++) {
        for (int j = 0; j < N; j++) {
            printf("stream %u, h_a[%u]: %u, h_b[%u]: %u.\n", s, j, h_a[s][j], j, h_b[s][j]);
        }
    }      

    // Expected behavior if fast_first = 1

    // iter from 0 to 4.

    bool fast_first = 1;

    //for (int iter = 0; iter < num_streams * num_iter; ++iter) {
    for (int i = 0; i < num_iter; ++i) {
        for (int s = 0; s < num_streams; ++s) {

        // i = iterate through streams within each loop.

        int i1 = i % num_streams;
    
        // make async copy for current stream.

        cudaMemcpyAsync(d_a[s], h_a[s], N, cudaMemcpyHostToDevice, streams[s]);
        cudaMemcpyAsync(d_b[s], h_b[s], N, cudaMemcpyHostToDevice, streams[s]);
        cudaMemcpyAsync(d_c[s], h_c[s], N, cudaMemcpyHostToDevice, streams[s]);

        // i = 0, slow launches first or 3rd. i=1, fast launches second and 4th.

        printf("iter: %u.\n", i);

        int threads=256;
        int blocks=N/threads + 1;

        switch(s) {
            case 0:
                printf("Launching slow stream i: %u\n", i);
                slow<<<blocks, threads, 0, streams[s]>>>(d_a[s], d_b[s], d_c[s]);
                break;
            case 1:
                printf("Launching fast stream %u.\n", i);
                fast<<<blocks, threads, 0, streams[s]>>>(d_a[s], d_b[s], d_c[s]);
                break;
            case 2:
                printf("Launching slow2x stream %u.\n", i);
                slow2x<<<blocks, threads, 0, streams[s]>>>(d_a[s], d_b[s], d_c[s]);
                break;
            case 3:
                printf("Launching fast2x stream %u.\n", i);
                fast2x<<<blocks, threads, 0, streams[s]>>>(d_a[s], d_b[s], d_c[s]);
                break;
            default:
                printf("Bypassing stream %u.\n", i);
            }
        // copy back.
        
        cudaMemcpyAsync(h_a[s], d_a[s], N, cudaMemcpyDeviceToHost, streams[s]);
        cudaMemcpyAsync(h_b[s], d_b[s], N, cudaMemcpyDeviceToHost, streams[s]);
        cudaMemcpyAsync(h_c[s], d_c[s], N, cudaMemcpyDeviceToHost, streams[s]);
        }
    }

    cudaDeviceSynchronize();

    for (int s = 0; s < num_streams; s++) {
        for (int j = 0; j < N; j++) {
            printf("stream=%u. idx=%u: %u + %u = %u.\n", s, j, h_a[s][j], h_b[s][j], h_c[s][j]);
        }
    }      


    for (int s = 0; s < num_streams; s++) {
        cudaStreamDestroy(streams[s]);
        cudaFree(d_a[s]);
    }
}
