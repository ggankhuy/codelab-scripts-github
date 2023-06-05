#include "hip/hip_runtime.h"
#include <stdio.h>
//#include "lib1/lib1.h"

__global__ void checkIndex(int * pKernelResp) {
    printf("threadIdx: (%d), %d, %d) blockIdx: (%d, %d, %d) blockDim (%d, %d, %d) gridDim: (%d, %d, %d)\n", \
        hipThreadIdx_x, hipThreadIdx_y, hipThreadIdx_z, \
        hipBlockIdx_x, hipBlockIdx_y, hipBlockIdx_z, \
        hipBlockDim_x, hipBlockDim_y, hipBlockDim_z, \
        hipGridDim_x, hipGridDim_y, hipGridDim_z); 
        *(pKernelResp+0)=hipThreadIdx_x;
        *(pKernelResp+1)=hipThreadIdx_y;
        *(pKernelResp+2)=hipThreadIdx_z;

        *(pKernelResp+3)=hipBlockIdx_x;
        *(pKernelResp+3)=hipBlockIdx_y;
        *(pKernelResp+3)=hipBlockIdx_z;

        *(pKernelResp+6)=hipBlockDim_x;
        *(pKernelResp+7)=hipBlockDim_y;
        *(pKernelResp+8)=hipBlockDim_z;

        *(pKernelResp+9)=hipGridDim_x;
        *(pKernelResp+10)=hipGridDim_y;
        *(pKernelResp+11)=hipGridDim_z;

        for (int i  = 0 ; i < 12 ; i ++) {
            *(pKernelResp+i) = i * 2;
        }
}

int main(int argc, char **argv) {

    // define total data element.

    int nElem = 6;
    int kernelRes[9] = {0,0,0, 0,0,0, 0,0,0};

    // define grid and block structure.

    dim3 block(3, 2, 5);
    dim3 grid ((nElem + block.x-1)/block.x);

    // check grid and block dimension from host side

    printf("grid.x %d grid.y %d grid.z %d\n", grid.x, grid.y, grid.z);
    printf("block.x %d block.y %d block.z %d\n", block.x, block.y, block.z);

    // check grid and block dimension from device side

    checkIndex <<<grid,block>>>(kernelRes);
    /*printf("threadIdx: (%d, %d, %d) blockIdx: (%d, %d, %d) blockDim (%d, %d, %d) gridDim: (%d, %d, %d)\n", \
        *(kernelRes+0), *(kernelRes+1), *(kernelRes+2), \
        *(kernelRes+3), *(kernelRes+4), *(kernelRes+5), \
        *(kernelRes+6), *(kernelRes+7), *(kernelRes+8), \
        *(kernelRes+9), *(kernelRes+10), *(kernelRes+11));*/
    hipDeviceReset();
    return 0;
}
