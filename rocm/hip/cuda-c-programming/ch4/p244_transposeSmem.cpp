#include <iostream>
#include <sys/time.h>
#include <lib.h>
#include <kernels.h>
#include <hip/hip_runtime.h>

void usage() {
    printf("Usage: ");
    printf("<execname> <p1> <p2> <p3> <p4> where: \n");
    printf("p1 p2: blockx, blocky.\n");
    printf("p3 p4: nx, ny.\n");
    return;
}

int main(int argc, char **argv) {
    // setup a device.

    int dev = 0;
    hipDeviceProp_t deviceProp;
    hipGetDeviceProperties(&deviceProp, dev);
    printf("%s starting transpose at ", argv[0]);
    printf("device %d: %s", dev, deviceProp.name);
    hipSetDevice(dev);

    // setup array size 2048

    int nx = 1<<11;
    int ny = 1<<11;

    // select a kernel and block size.

    int iKernel=0;
    int blockx=16;
    int blocky=16;

    if (argc == 1) {usage(); return 1;};
    //if (argc > 1) iKernel = atoi(argv[1]);
    if (argc > 1) blockx = atoi(argv[1]);
    if (argc > 2) blocky = atoi(argv[2]);
    if (argc > 3) nx = atoi(argv[3]);
    if (argc > 4) ny = atoi(argv[4]);

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
    hipMalloc((float**)&d_A, nBytes);
    hipMalloc((float**)&d_C, nBytes);

    // copy data from host to device

    hipMemcpy(d_A, h_A, nBytes, hipMemcpyHostToDevice);

    // warmup to avoid start overhead.

    double iStart = seconds();
    warmup <<< grid, block >>> (d_C, d_A, nx, ny);
    hipDeviceSynchronize();
    double iElaps = seconds() - iStart;
    printf("warmup elapsed %f sec.\n", iElaps);
    
    // kernel pointer and descriptor.

    void (*kernel)(float *, float*, int, int);
    char *kernelName;

    // setup a kernel.

    /*
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
    */

    // run kernel.

    iStart = seconds();
    transposeSmem<<<grid, block>>>(d_C, d_A, nx, ny);
    hipDeviceSynchronize();
    iElaps = seconds() - iStart;

    // calculate eff. bw.

    float ibnd = 2*nx*ny*sizeof(float)/1e9/iElaps;
    printf("%s elapsed %f sec <<< grid (%d,%d) block (%d,%d) >>> effective bw: %f GB\n", kernelName, iElaps, grid.x, grid.y, block.x, block.y, ibnd);
    
    // check kernel results. skipping.

    /*if (iKernel>1) {
        hipMemcpy(gpuRef, d_C, nBytes, hipMemcpyDeviceToHost);
    }*/
    
    // free host and device memory.

    hipFree(d_A);
    hipFree(d_C);
    free(h_A);
    free(gpuRef);

    return 0;


}








