//#include "../common/common.h"
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <lib.h>
#include <kernels.h>

/*
 * An example of using shared memory to transpose square thread coordinates
 * of a hip grid into a global memory array. Different kernels below
 * demonstrate performing reads and writes with different ordering, as well as
 * optimizing using memory padding.
 */

#define BDIMX 32
#define BDIMY 32
#define IPAD  1

void printData(char *msg, int *in,  const int size)
{
    printf("%s: ", msg);

    for (int i = 0; i < size; i++)
    {
        printf("%5d", in[i]);
        fflush(stdout);
    }

    printf("\n");
    return;
}

__global__ void setRowReadRow (int *out)
{
    // static shared memory
    __shared__ int tile[BDIMY][BDIMX];

    // mapping from thread index to global memory index
    unsigned int idx = hipThreadIdx_y * hipBlockDim_x + hipThreadIdx_x;

    // shared memory store operation
    tile[hipThreadIdx_y][hipThreadIdx_x] = idx;

    // wait for all threads to complete
    __syncthreads();

    // shared memory load operation
    out[idx] = tile[hipThreadIdx_y][hipThreadIdx_x] ;
}

__global__ void setColReadCol (int *out)
{
    // static shared memory
    __shared__ int tile[BDIMX][BDIMY];

    // mapping from thread index to global memory index
    unsigned int idx = hipThreadIdx_y * hipBlockDim_x + hipThreadIdx_x;

    // shared memory store operation
    tile[hipThreadIdx_x][hipThreadIdx_y] = idx;

    // wait for all threads to complete
    __syncthreads();

    // shared memory load operation
    out[idx] = tile[hipThreadIdx_x][hipThreadIdx_y];
}

__global__ void setRowReadCol(int *out)
{
    // static shared memory
    __shared__ int tile[BDIMY][BDIMX];

    // mapping from thread index to global memory index
    unsigned int idx = hipThreadIdx_y * hipBlockDim_x + hipThreadIdx_x;

    // shared memory store operation
    tile[hipThreadIdx_y][hipThreadIdx_x] = idx;

    // wait for all threads to complete
    __syncthreads();

    // shared memory load operation
    out[idx] = tile[hipThreadIdx_x][hipThreadIdx_y];
}


__global__ void setRowReadColDyn(int *out)
{
    // dynamic shared memory
    extern  __shared__ int tile[];

    // mapping from thread index to global memory index
    unsigned int row_idx = hipThreadIdx_y * hipBlockDim_x + hipThreadIdx_x;
    unsigned int col_idx = hipThreadIdx_x * hipBlockDim_y + hipThreadIdx_y;

    // shared memory store operation
    tile[row_idx] = row_idx;

    // wait for all threads to complete
    __syncthreads();

    // shared memory load operation
    out[row_idx] = tile[col_idx];
}

__global__ void setRowReadColPad(int *out)
{
    // static shared memory
    __shared__ int tile[BDIMY][BDIMX + IPAD];

    // mapping from thread index to global memory offset
    unsigned int idx = hipThreadIdx_y * hipBlockDim_x + hipThreadIdx_x;

    // shared memory store operation
    tile[hipThreadIdx_y][hipThreadIdx_x] = idx;

    // wait for all threads to complete
    __syncthreads();

    // shared memory load operation
    out[idx] = tile[hipThreadIdx_x][hipThreadIdx_y];
}

__global__ void setRowReadColDynPad(int *out)
{
    // dynamic shared memory
    extern  __shared__ int tile[];

    // mapping from thread index to global memory index
    unsigned int row_idx = hipThreadIdx_y * (hipBlockDim_x + IPAD) + hipThreadIdx_x;
    unsigned int col_idx = hipThreadIdx_x * (hipBlockDim_x + IPAD) + hipThreadIdx_y;

    unsigned int g_idx = hipThreadIdx_y * hipBlockDim_x + hipThreadIdx_x;

    // shared memory store operation
    tile[row_idx] = g_idx;

    // wait for all threads to complete
    __syncthreads();

    // shared memory load operation
    out[g_idx] = tile[col_idx];
}

int main(int argc, char **argv)
{
    // set up device
    int dev = 0;
    hipDeviceProp_t deviceProp;
    CHECK(hipGetDeviceProperties(&deviceProp, dev));
    printf("%s at ", argv[0]);
    printf("device %d: %s ", dev, deviceProp.name);
    CHECK(hipSetDevice(dev));

    hipSharedMemConfig pConfig;
    CHECK(hipDeviceGetSharedMemConfig ( &pConfig ));
    printf("with Bank Mode:%s ", pConfig == 1 ? "4-Byte" : "8-Byte");

    // set up array size 2048
    int nx = BDIMX;
    int ny = BDIMY;

    bool iprintf = 0;

    if (argc > 1) iprintf = atoi(argv[1]);

    size_t nBytes = nx * ny * sizeof(int);

    // execution configuration
    dim3 block (BDIMX, BDIMY);
    dim3 grid  (1, 1);
    printf("<<< grid (%d,%d) block (%d,%d)>>>\n", grid.x, grid.y, block.x,
           block.y);

    // allocate device memory
    int *d_C;
    CHECK(hipMalloc((int**)&d_C, nBytes));
    int *gpuRef  = (int *)malloc(nBytes);

    CHECK(hipMemset(d_C, 0, nBytes));
    setColReadCol<<<grid, block>>>(d_C);
    CHECK(hipMemcpy(gpuRef, d_C, nBytes, hipMemcpyDeviceToHost));

    if(iprintf)  printData("set col read col   ", gpuRef, nx * ny);

    CHECK(hipMemset(d_C, 0, nBytes));
    setRowReadRow<<<grid, block>>>(d_C);
    CHECK(hipMemcpy(gpuRef, d_C, nBytes, hipMemcpyDeviceToHost));

    if(iprintf)  printData("set row read row   ", gpuRef, nx * ny);

    CHECK(hipMemset(d_C, 0, nBytes));
    setRowReadCol<<<grid, block>>>(d_C);
    CHECK(hipMemcpy(gpuRef, d_C, nBytes, hipMemcpyDeviceToHost));

    if(iprintf)  printData("set row read col   ", gpuRef, nx * ny);

    CHECK(hipMemset(d_C, 0, nBytes));
    setRowReadColDyn<<<grid, block, BDIMX*BDIMY*sizeof(int)>>>(d_C);
    CHECK(hipMemcpy(gpuRef, d_C, nBytes, hipMemcpyDeviceToHost));

    if(iprintf)  printData("set row read col dyn", gpuRef, nx * ny);

    CHECK(hipMemset(d_C, 0, nBytes));
    setRowReadColPad<<<grid, block>>>(d_C);
    CHECK(hipMemcpy(gpuRef, d_C, nBytes, hipMemcpyDeviceToHost));

    if(iprintf)  printData("set row read col pad", gpuRef, nx * ny);

    CHECK(hipMemset(d_C, 0, nBytes));
    setRowReadColDynPad<<<grid, block, (BDIMX + IPAD)*BDIMY*sizeof(int)>>>(d_C);
    CHECK(hipMemcpy(gpuRef, d_C, nBytes, hipMemcpyDeviceToHost));

    if(iprintf)  printData("set row read col DP ", gpuRef, nx * ny);

    // free host and device memory
    CHECK(hipFree(d_C));
    free(gpuRef);

    // reset device
    CHECK(hipDeviceReset());
    return EXIT_SUCCESS;
}
