
#include <stdio.h>
#include <stdbool.h>
#include <cuda_runtime.h>
#include <cuda.h>
#define ALLOC_NORMAL 1
#define ALLOC_PAGE_LOCKED 2
#define SIZE (64*1024*1024) // 10MB.

float cuda_host_alloc_test(int size, bool up, int allocType) {
    int debug = 0;
    cudaEvent_t start, stop;
    int *a, *dev_a;
    float elapsedTime;

    cudaEventCreate( &start);
    cudaEventCreate( &stop);

    if (allocType == ALLOC_PAGE_LOCKED) {
        cudaHostAlloc((void**)&a, size * sizeof(*dev_a), cudaHostAllocDefault);
    } else if  (allocType == ALLOC_NORMAL ) {
        a = (int*) malloc(size*sizeof(*a));
    }

    cudaMalloc((void**)&dev_a, size * sizeof(*dev_a));

    cudaEventRecord( start, 0);
    for (int i = 0; i < 100 ; i ++ ) {
        if (up) {  
            cudaMemcpy(dev_a, a, size * sizeof(*a), cudaMemcpyHostToDevice);
        } else { 
            cudaMemcpy(a, dev_a, size * sizeof(*a), cudaMemcpyDeviceToHost);
        }
               
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime( &elapsedTime, start, stop);

    if (allocType == ALLOC_PAGE_LOCKED) {
        cudaFreeHost(a);
    } else if  (allocType == ALLOC_NORMAL ) {
        free(a);
    }
    cudaFree(dev_a);
    return elapsedTime;
}
int main()
{
    float elapsedTime;

    float MB  = (float)100 * SIZE * sizeof(int) / 1024 / 1024;

    elapsedTime = cuda_host_alloc_test(SIZE, true, ALLOC_PAGE_LOCKED);
    printf("Time using cudaHostAlloc: %3.1f ms.\n", elapsedTime);
    printf("\tMB/s during copy up: %3.1f.\n", MB / (elapsedTime / 1000));

    elapsedTime = cuda_host_alloc_test(SIZE, false, ALLOC_PAGE_LOCKED);
    printf("Time using cudaHostAlloc: %3.1f ms.\n", elapsedTime);
    printf("\tMB/s during copy down: %3.1f.\n", MB / (elapsedTime / 1000));
    
    elapsedTime = cuda_host_alloc_test(SIZE, true, ALLOC_NORMAL);
    printf("Time using cudaMalloc: %3.1f ms.\n", elapsedTime);
    printf("\tMB/s during copy up: %3.1f.\n", MB / (elapsedTime / 1000));

    elapsedTime = cuda_host_alloc_test(SIZE, false, ALLOC_NORMAL);
    printf("Time using cudaMalloc: %3.1f ms.\n", elapsedTime);
    printf("\tMB/s during copy down: %3.1f.\n", MB / (elapsedTime / 1000));

}
