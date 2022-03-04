#include <stdio.h>
#include "hip/hip_runtime.h"

#define imin(a,b) (a<b?a:b)

const int N = 64;
const int threadsPerBlock = 1;
const int blocksPerGrid = imin (32, (N+threadsPerBlock-1)/(threadsPerBlock));

__global__ void dot(float * a, float *b, float *c) {

    // arrSize = threads in block.

    __shared__ float arr1[threadsPerBlock];
    int gid = hipThreadIdx_x  + hipBlockIdx_x * hipBlockDim_x;
    int lid = hipThreadIdx_x;

    // accumulates a * b in every block.
    // note += a * b below.

    float product = 0;
     
    while ( gid < N) {
        product += a[gid] * b[gid];
        gid += hipBlockDim_x  * hipGridDim_x;
    }
    arr1[lid] = product;
    __syncthreads();

    // reductions.
    
    int i =  hipBlockDim_x/2;
    while (i != 0 ) {
        if (lid < i) {
            arr1[lid] += arr1[lid + i];
        }
        __syncthreads();
        i /= 2;
    }
    if (lid == 0)
        c[hipBlockIdx_x] = arr1[0];
}

int main( void ) {

    // partial c holds array len of block size.

    float *a, *b, c, *partial_c;
    float *dev_a, *dev_b, *dev_c, *dev_partial_c;
 
    a = (float*) malloc(N * sizeof(float));
    b = (float*) malloc(N * sizeof(float));
    partial_c = (float *) malloc(blocksPerGrid * sizeof(float));

    hipMalloc((void**)&dev_a, N * sizeof(float));
    hipMalloc((void**)&dev_b, N * sizeof(float));
    hipMalloc((void**)&dev_partial_c, blocksPerGrid * sizeof(float));

    for (int i = 0; i < N ; i++) {
        a[i] = i;
        b[i] = i*2;
    }

    hipMemcpy(dev_a, a, N*sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(dev_b, b, N*sizeof(float), hipMemcpyHostToDevice);

    printf("blocksPerGrid, threadsPerBlock: %u, %u.\n", blocksPerGrid, threadsPerBlock);
    dot <<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_partial_c);
    hipMemcpy(partial_c, dev_partial_c, blocksPerGrid*sizeof(float), hipMemcpyDeviceToHost);

    c = 0;
    for (int i = 0; i < blocksPerGrid; i++) {
        c += partial_c[i];
        //printf("partial_c:%x /c: %d/%d.\n", i, partial_c[i], c); 
        printf("partial_c[$i], %g, c: %g.\n", i, partial_c[i], c); 
    }

    printf("partial_c:%.6g.\n", c); 

    #define sum_squares(x) (x*(x+1)*(2*x+1)/6)
    printf("Does gpu value %.6g= %.6g?\n", c, 2 * sum_squares((float) (N-1)));
    
    hipFree(dev_a);
    hipFree(dev_b);
    hipFree(dev_partial_c);
    
    free(a);
    free(b);
    free(partial_c);
}
