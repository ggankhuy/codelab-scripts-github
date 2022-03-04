/* doc vector not using shared memory for comparison. WIP */ 

#include <stdio.h>
#include "hip/hip_runtime.h"

#define imin(a,b) (a<b?a:b)

const int N = 8192;
const int threadsPerBlock = 256;
const int blocksPerGrid = imin (32, (N+threadsPerBlock-1)/(threadsPerBlock));

__global__ void dot(float * a, float *b, float *c) {

    // arrSize = threads in block.

    int gid = hipThreadIdx_x  + hipBlockIdx_x * hipBlockDim_x;
     
    while ( gid < N) {
        c[gid] = a[gid] * b[gid];
        gid += hipBlockDim_x  * hipGridDim_x;
    }

    int i =  N/2;
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

    partial_c = (float *) malloc(N * sizeof(float));

    hipMalloc((void**)&dev_a, N * sizeof(float));
    hipMalloc((void**)&dev_b, N * sizeof(float));
    hipMalloc((void**)&dev_partial_c, blocksPerGrid * sizeof(float));

    for (int i = 0; i < N ; i++) {
        a[i] = i;
        b[i] = i*2;
    }

    hipMemcpy(dev_a, a, N*sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(dev_b, b, N*sizeof(float), hipMemcpyHostToDevice);

    dot <<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_partial_c);
    hipMemcpy(partial_c, dev_partial_c, blocksPerGrid*sizeof(float), hipMemcpyDeviceToHost);

    c = 0;
    for (int i = 0; i < blocksPerGrid; i++) {
        c += partial_c[i];
    }

    #define sum_squares(x) (x*(x+1)*(2*x+1)/6)
    printf("Does gpu value %.6g= %.6g?\n", c, 2 * sum_squares((float) (N-1)));
    
    hipFree(dev_a);
    hipFree(dev_b);
    hipFree(dev_partial_c);
    
    free(a);
    free(b);
    free(partial_c);
}
