/*
Copyright (c) 2015 - 2021 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <stdio.h>
#include "hip/hip_runtime.h"
#include "vector.h"
#include <iostream>
#include <string.h>

#define WIDTH_X 16
#define WIDTH_Y 16

#define NUM (WIDTH_X * WIDTH_Y)

#define THREADS_X 4
#define THREADS_Y 4
#define THREADS_Z 1

using namespace std;

// Device (Kernel) function, it must be void

/*
__global__ void matrixTranspose(float* out, float* in, const int width) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    out[y * width + x] = in[x * width + y];
}
*/

__global__ void add(float* a, float* b, float *c, const int nx, const int ny) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    
    if (x < nx && y < ny)
        c[y * nx + x] = a[y * nx + x] + b[y * nx + x];
}

int main() {
    float* matrixA;
    float* matrixB;
    float* matrixC;
    float* gpuMatrixA;
    float* gpuMatrixB; 
    float* gpuMatrixC;;
    int LOOPSTRIDE = 1;

    /*
    char* env_project_name;
    string env_project_name_str = "";
    env_project_name=std::getenv("PROJECT_NAME");
    env_project_name ? env_project_name_str=string(env_project_name): "" ;

    if (env_project_name_str == "matrix1024") { NUM = 4; LOOPSTRIDE=1; }
    */
    hipDeviceProp_t devProp;
    hipGetDeviceProperties(&devProp, 0);

    std::cout << "Device name " << devProp.name << std::endl;

    int i;
    int errors;

    matrixA = (float*)malloc(NUM * sizeof(float));
    matrixB = (float*)malloc(NUM * sizeof(float));
    matrixC = (float*)malloc(NUM * sizeof(float));

    // initialize the input data
    for (i = 0; i < NUM; i++) {
        matrixA[i] = (float)i * 1.0f;
        matrixA[i] = (float)i * 2.0f;
        matrixA[i] = 999.0f;
    }

    // allocate the memory on the device side
    hipMalloc((void**)&gpuMatrixA, NUM * sizeof(float));
    hipMalloc((void**)&gpuMatrixB, NUM * sizeof(float));
    hipMalloc((void**)&gpuMatrixC, NUM * sizeof(float));

    // Memory transfer from host to device
    hipMemcpy(gpuMatrixA, matrixA, NUM * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(gpuMatrixB, matrixB, NUM * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(gpuMatrixC, matrixC, NUM * sizeof(float), hipMemcpyHostToDevice);

    // Lauching kernel from host
    add<<<dim3(WIDTH_X / THREADS_X, WIDTH_Y / THREADS_Y),  dim3(THREADS_X, THREADS_Y)>>>(gpuMatrixA, gpuMatrixB, gpuMatrixC, WIDTH_X, WIDTH_Y);
    //matrixTranspose<<<dim3(WIDTH / THREADS_X, WIDTH / THREADS_Y),  dim3(THREADS_X, THREADS_Y)>>>(gpuMatrixC, gpuMatrixA, WIDTH);
    /*hipLaunchKernelGGL(matrixTranspose, dim3(WIDTH / THREADS_X, WIDTH / THREADS_Y),
                    dim3(THREADS_X, THREADS_Y), 0, 0, gpuMatrixC,
                    gpuMatrixA, WIDTH);
    */

    // Memory transfer from device to host
 
    hipMemcpy(matrixA, gpuMatrixA, NUM * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(matrixB, gpuMatrixB, NUM * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(matrixC, gpuMatrixC, NUM * sizeof(float), hipMemcpyDeviceToHost);

    // verify the results

    // free the resources on device side

    hipFree(gpuMatrixA);
    hipFree(gpuMatrixB);
    hipFree(gpuMatrixC);

    // free the resources on host side

    free(matrixA);
    free(matrixB);
    free(matrixC);

    return errors;
}
