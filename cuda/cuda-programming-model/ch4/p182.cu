#include <iostream>
#include <sys/time.h>
#include <lib.h>
#include <kernels.h>

void usage() {
    printf("Usage: ");
    printf("<execname> <p1> <p2> <p3> <p4> where: \n");
    printf("p1: kernel name: \n");
    printf(" - 0:copyrow\n - 1:copyCol\n - 2: transposeRow\n - 3: transposeCol.\n");
    printf("p2 p3: blockx, blocky.\n");
    printf("p4 p5: nx, ny.\n");
    return;
}

__global__ void copyRow(float * out, float * in, const int nx, const int ny) {
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < nx && iy < ny) {
        out[iy * nx + ix] = in[iy * nx + ix];
    }
}

__global__ void copyCol(float * out, float * in, const int nx, const int ny) {
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < nx && iy < ny) {
        out[ix * ny + iy] = in[ix * ny + iy];
    }
}

__global__ void transposeNaiveRow(float * out, float * in, const int nx, const int ny) {
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < nx && iy < ny) {
        out[ix*ny + iy] = in[iy * nx + ix];
    }
}

__global__ void transposeNaiveCol(float * out, float * in, const int nx, const int ny) {
    unsigned int ix=blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy=blockDim.y * blockIdx.y + threadIdx.y;
    if (ix<nx && iy<ny) {
        out[iy * nx + ix] = in[ix * ny + iy];
    }
}
__global__ void warmup(float * out, float * in, const int nx, const int ny) {
    unsigned int ix=blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy=blockDim.y * blockIdx.y + threadIdx.y;
    if (ix<nx && iy<ny) {
        out[iy * nx + ix] = in[ix * ny + iy];
    }
} 

int main(int argc, char **argv) {
    // setup a device.

    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("%s starting transpose at ", argv[0]);
    printf("device %d: %s", dev, deviceProp.name);
    cudaSetDevice(dev);

    // setup array size 2048

    int nx = 1<<11;
    int ny = 1<<11;

    // select a kernel and block size.

    int iKernel=0;
    int blockx=16;
    int blocky=16;

    if (argc == 1) {usage(); return 1;};
    if (argc > 1) iKernel = atoi(argv[1]);
    if (argc > 2) blockx = atoi(argv[2]);
    if (argc > 3) blocky = atoi(argv[3]);
    if (argc > 4) nx = atoi(argv[4]);
    if (argc > 5) ny = atoi(argv[5]);

    printf(" with matrix nx %d ny %d with kernel %d.\n", nx, ny, iKernel);
    size_t nBytes  = nx * ny * sizeof(float);
    
    // execution configuration

    dim3 block(blockx, blocky);
    dim3 grid((nx+block.x-1)/block.x, (ny+block.y-1)/block.y);

    // allocate host memory.

    float *h_A = (float*)malloc(nBytes);
    float *hostRef = (float*)malloc(nBytes);
    float *gpuRef = (float*)malloc(nBytes);

    // init host array.

    initialData(h_A, nx*ny);
    
    // transport at host (skipping).

    // allocate device memory.

    float *d_A, *d_C;
    cudaMalloc((float**)&d_A, nBytes);
    cudaMalloc((float**)&d_C, nBytes);

    // copy data from host to device

    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);

    // warmup to avoid start overhead.

    double iStart = seconds();
    warmup <<< grid, block >>> (d_C, d_A, nx, ny);
    cudaDeviceSynchronize();
    double iElaps = seconds() - iStart;
    printf("warmup elapsed %f sec.\n", iElaps);
    
    // kernel pointer and descriptor.

    void (*kernel)(float *, float*, int, int);
    char *kernelName;

    // setup a kernel.

    switch(iKernel) {
        case 0:
            kernel=&copyRow;
            kernelName = "copyRow    ";
            break;
        case 1:
            kernel=&copyCol;
            kernelName = "copyCol    ";
            break;
        case 2:
            kernel=&transposeNaiveRow;
            kernelName = "NaiveRow    ";
            break;
        case 3:
            kernel=&transposeNaiveCol;
            kernelName = "NaiveCol     ";
            break;
    }

    // run kernel.

    iStart = seconds();
    kernel <<<grid, block>>>(d_C, d_A, nx, ny);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;

    // calculate eff. bw.

    float ibnd = 2*nx*ny*sizeof(float)/1e9/iElaps;
    printf("%s elapsed %f sec <<< grid (%d,%d) block (%d,%d) >>> effective bw: %f GB\n", kernelName, iElaps, grid.x, grid.y, block.x, block.y, ibnd);
    
    // check kernel results. skipping.

    /*if (iKernel>1) {
        cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
    }*/
    
    // free host and device memory.

    cudaFree(d_A);
    cudaFree(d_C);
    free(h_A);
    free(gpuRef);

    return 0;


}








