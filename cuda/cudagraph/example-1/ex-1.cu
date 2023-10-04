// https://developer.nvidia.com/blog/cuda-graphs/
//#include <cuda_runtime_api.h>
#include <stdio.h>

#define N 500000 // tuned such that kernel takes a few microseconds
#define NSTEP 1000
#define NKERNEL 20

__global__ void shortKernel(float * out_d, float * in_d){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if(idx<N) 
        out_d[idx] = 1.23 * in_d[idx];
}

// start CPU wallclock timer

int main () {

    float *out_d, *in_d;
    float *out_h, *in_h;

    cudaMalloc( (void**)&in_d, N * sizeof(float) );
    cudaMalloc( (void**)&out_d, N * sizeof(float) );

    out_h = (float*)malloc( N * sizeof(float));
    in_h = (float*)malloc( N * sizeof(float));

    for (int i = 0; i < N ; i ++ ) {
            in_h[i] = i + 200;
    }
    cudaMemcpy(in_d, in_h, N * sizeof(float), cudaMemcpyHostToDevice);

    cudaStream_t    stream;
    cudaStreamCreate( &stream );

    for(int istep=0; istep<NSTEP; istep++){
        for(int ikrnl=0; ikrnl<NKERNEL; ikrnl++){
            shortKernel<<<1024, 1, 0, stream>>>(out_d, in_d);
            cudaStreamSynchronize(stream);
        }
    }
}
//end CPU wallclock time
