/*zero copy memory example.
cudahostAlloc()
On normal: both host and gpu has separate malloc, cudamalloc.
On zero-copy: only alloc memory on host and get device pointer.
*/

#include <stdio.h>
#include <stdbool.h>
#include <cuda_runtime.h>
#include <cuda.h>
#define ALLOC_NORMAL 1
#define ALLOC_PAGE_HOST 3
#define imin(a,b) (a<b?a:b)

const int N = 33 * 1024 * 1024;
const int threadsPerBlock = 256;
const int blocksPerGrid = imin(32, (N + threadsPerBlock - 1 ) / threadsPerBlock) ;

__global__ void dot(float * a, float * b, float * c) {

    // this variable will be repeated and will have same value on every blocks.

    __shared__ float cache[threadsPerBlock];

    int tid = threadIdx.x * blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    float temp = 0;
    while (tid < N)  {

        // temp will accumulate the product of a and b at every at every grid.

        temp += a[tid] * b[tid] ;
        tid += blockDim.x * gridDim.x;
    }

    // holds product of a and b at every grid.

    cache[cacheIndex] = temp;

    // sync threads in this block.

    __syncthreads();

    // for reductions, threadsPerBlock must be a power of 2 because of following code.
    // couple of examples (still can not understand the divide in loop.
    // blockDim.x = 256.
    //                              i         cache[tid] + cache[tid+i]
    // threadIdx/cacheindex 128:    i0 = 128, cache[128] + cache[128+128] -> t128, t256.
    // threadIdx/cacheindex 127:    i0 = 128, cache[127] + cache[128+127] -> t127, t255.
    // threadIdx/cacheindex 126:    i0 = 128, cache[126] + cache[128+126] -> t126, t254.
    // threadIdx/cacheindex 125:    i0 = 128, cache[125] + cache[128+125] -> t125, t253.
    // ...
    // threadIdx/cacheindex 64:     i0 = 128, cache[64] + cache[128+64] -> t64, t192.
    // ...
    // threadIdx/cacheindex 32:     i0 = 128, cache[32] + cache[128+32] -> t32, t160.
    // ...
    // threadIdx/cacheindex 2:      i0 = 128, cache[2] + cache[128+2] -> t2, t130.
    // threadIdx/cacheindex 1:      i0 = 128, cache[1] + cache[128+1] -> t1, t129.

    int i = blockDim.x/2;

    while (i != 0 ) {
        if (cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];

        __syncthreads();
        i /= 2 ;
    }
    if (cacheIndex == 0) {
        c[blockIdx.x] = cache[0];
    }
}

float alloc_test(int size, int allocType) {
    int debug = 0;
    cudaEvent_t start, stop;
    float *a, *b, c, *partial_c, *dev_a, *dev_b, *dev_partial_c;
    float elapsedTime;

    cudaEventCreate( &start);
    cudaEventCreate( &stop);

    if (allocType == ALLOC_PAGE_HOST) {
        cudaHostAlloc((void**)&a, size * sizeof(float), cudaHostAllocWriteCombined|cudaHostAllocMapped);
        cudaHostAlloc((void**)&b, size * sizeof(float), cudaHostAllocWriteCombined|cudaHostAllocMapped);
        cudaHostAlloc((void**)&partial_c, blocksPerGrid * sizeof(float), cudaHostAllocMapped);
    } else if  (allocType == ALLOC_NORMAL ) {
        a = (float*) malloc(size * sizeof(float));
        b = (float*) malloc(size * sizeof(float));
        partial_c = (float*) malloc(size * sizeof(float));
        cudaMalloc((void**)&dev_a, size * sizeof(float));
        cudaMalloc((void**)&dev_b, size * sizeof(float));
        cudaMalloc((void**)&dev_partial_c, blocksPerGrid * sizeof(float));
    }

    for (int i = 0; i < size ; i++ ) {
        a[i] = i;
        b[i] = i * 2;
    }

    if (allocType == ALLOC_PAGE_HOST) {
        cudaHostGetDevicePointer(&dev_a, a, 0);
        cudaHostGetDevicePointer(&dev_b, b, 0);
        cudaHostGetDevicePointer(&dev_partial_c, partial_c, 0);
    }

    cudaEventRecord( start, 0);

    if  (allocType == ALLOC_NORMAL ) {
        cudaMemcpy(dev_a, a, size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_b, b, size * sizeof(float), cudaMemcpyHostToDevice);
    }

    dot<<<blocksPerGrid, threadsPerBlock>>>( dev_a, dev_b, dev_partial_c);

    if  (allocType == ALLOC_NORMAL ) {
        cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);    
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime( &elapsedTime, start, stop);

    c = 0;
    for (int i = 0; i < blocksPerGrid; i++)
        c += partial_c[i];

    #define sum_squares(x) (x*(x+1)*(2*x+1)/6)

    printf("Does GPU value %.6g = %.6g?\n", c, 2 * sum_squares((float) (N-1)));

    // free memory on gpu side.

    if (allocType == ALLOC_PAGE_HOST) {
        cudaFreeHost(dev_a);
        cudaFreeHost(dev_b);
        cudaFreeHost(dev_partial_c);
    } else if (allocType == ALLOC_NORMAL) {
        cudaFree(dev_a);
        cudaFree(dev_b);
        cudaFree(dev_partial_c);

        // free on cpu side.

        free(a);
        free(b);
        free(partial_c);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("Value calculated: %04.02f.\n", c);

    return elapsedTime;
}
int main()
{
    float elapsedTime;
    cudaDeviceProp prop;
    int whichDevice;
    cudaGetDevice(&whichDevice);
    cudaGetDeviceProperties(&prop, whichDevice);

    if (prop.canMapHostMemory != 1 ) {
        printf("Device can not map host memory.\n");
        return 0;
    } else {
        printf("Device can map host memory ok...\n");
    }
    cudaSetDeviceFlags( cudaDeviceMapHost);
    printf("set flag to map host memory...\n");

    //float MB  = (float)100 * SIZE * sizeof(int) / 1024 / 1024;
    printf("Calling alloc_test with ALLOC_NORMAL...\n");

    elapsedTime = alloc_test(N, ALLOC_NORMAL);
    printf("Time using cudaMalloc: %3.1f ms.\n", elapsedTime);
    //printf("\tMB/s during copy up: %3.1f.\n", MB / (elapsedTime / 1000));

    printf("Calling alloc_test with ALLOC_PAGE_HOST...\n");
    elapsedTime = alloc_test(N, ALLOC_PAGE_HOST);
    printf("Time using cudaHostAlloc: %3.1f ms.\n", elapsedTime);
    //printf("\tMB/s during copy up: %3.1f.\n", MB / (elapsedTime / 1000));
    return 0;
}
