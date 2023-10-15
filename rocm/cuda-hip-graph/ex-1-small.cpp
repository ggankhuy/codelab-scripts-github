
#include "hip/hip_runtime.h"
#include <stdio.h>

#define N 500000 // tuned such that kernel takes a few microseconds

__global__ void shortKernel(float * out_d, float * in_d){
    int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (idx < N) 
        out_d[idx]=1.23*in_d[idx];
}

int main() {

    #define NSTEP 2
    #define NKERNEL 3

    float *out_d, *in_d;
    float *out_h, *in_h;

    hipMalloc( (void**)&in_d, N * sizeof(float) );
    hipMalloc( (void**)&out_d, N * sizeof(float) );

    out_h = (float*)malloc( N * sizeof(float));
    in_h = (float*)malloc( N * sizeof(float));

    for (int i = 0; i < N ; i ++ ) {
            in_h[i] = i + 200;
    }
    hipMemcpy(in_d, in_h, N * sizeof(float), hipMemcpyHostToDevice);

    hipStream_t    stream;
    hipStreamCreate( &stream );
    int blocks=512, threads=512;

    // start CPU wallclock timer
    for(int istep=0; istep<NSTEP; istep++) {
        for(int ikrnl=0; ikrnl<NKERNEL; ikrnl++) {
            shortKernel<<<blocks, threads, 0, stream>>>(out_d, in_d);
            hipStreamSynchronize(stream);
        }
    }
    //end CPU wallclock time

    return 0;    
}
