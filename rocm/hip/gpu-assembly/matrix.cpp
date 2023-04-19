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

#define N (WIDTH_X * WIDTH_Y)

#define THREADS_X 4
#define THREADS_Y 4
#define THREADS_Z 1

using namespace std;

// Device (Kernel) function, it must be void

/*
__global__ void matrixTranspose(int* out, int* in, const int width) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    out[y * width + x] = in[x * width + y];
}
*/

__global__ void add(int* a, int* b, int *c, const int nx, const int ny) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int tidx= y * nx + x;
    //if (x < nx && y < ny)
    c[tidx] = a[tidx] + b[tidx];
}

int main() {
    int* a;
    int* b;
    int* c;
    int* dev_a;
    int* dev_b; 
    int* dev_c;
    int LOOPSTRIDE = 4;

    /*
    char* env_project_name;
    string env_project_name_str = "";
    env_project_name=std::getenv("PROJECT_NAME");
    env_project_name ? env_project_name_str=string(env_project_name): "" ;

    if (env_project_name_str == "matrix1024") { N = 4; LOOPSTRIDE=1; }
    */
    /*
    hipDeviceProp_t devProp;
    hipGetDeviceProperties(&devProp, 0);

    std::cout << "Device name " << devProp.name << std::endl;
    */
    int i;
    int errors;

    a = (int*)malloc(N * sizeof(int));
    b = (int*)malloc(N * sizeof(int));
    c = (int*)malloc(N * sizeof(int));

    // initialize the input data
    for (i = 0; i < N; i++) {
        a[i] = (int)i * 5;
        b[i] = (int)i * 10;
        c[i] = 999;
    }

    // allocate the memory on the device side
    hipMalloc((void**)&dev_a, N * sizeof(int));
    hipMalloc((void**)&dev_b, N * sizeof(int));
    hipMalloc((void**)&dev_c, N * sizeof(int));

    // Memory transfer from host to device
    hipMemcpy(dev_a, a, N * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(dev_b, b, N * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(dev_c, c, N * sizeof(int), hipMemcpyHostToDevice);

    // Lauching kernel from host
    printf("<<<dim3(%u, %u), (%u, %u)>>>, widthx/y: %u, %u.\n", WIDTH_X / THREADS_X, WIDTH_Y / THREADS_Y, THREADS_X, THREADS_Y, WIDTH_X, WIDTH_Y);
    add<<<dim3(WIDTH_X / THREADS_X, WIDTH_Y / THREADS_Y),  dim3(THREADS_X, THREADS_Y)>>>(dev_a, dev_b, dev_c, WIDTH_X, WIDTH_Y);
    //matrixTranspose<<<dim3(WIDTH / THREADS_X, WIDTH / THREADS_Y),  dim3(THREADS_X, THREADS_Y)>>>(dev_c, dev_a, WIDTH);
    /*hipLaunchKernelGGL(matrixTranspose, dim3(WIDTH / THREADS_X, WIDTH / THREADS_Y),
                    dim3(THREADS_X, THREADS_Y), 0, 0, dev_c,
                    dev_a, WIDTH);
    */

    // Memory transfer from device to host
 
    hipMemcpy(a, dev_a, N * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(b, dev_b, N * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(c, dev_c, N * sizeof(int), hipMemcpyDeviceToHost);

    // verify the results

    for (int i = 0; i < N; i+=LOOPSTRIDE ) {
        printf("After add: %d: %u + %u = %u\n", i, a[i], b[i], c[i]);
        /*
        if (env_op_str == "add") { printf("After add: %d: %u + %u = %u\n", i, a[i], b[i], c[i]); }
        if (env_op_str == "mul") { printf("After mul: %d: %u * %u = %u\n", i, a[i], b[i], c[i]); }
        if (env_op_str == "mul_add") { printf("After mul_add: %d: %u * 2 + %u = %u\n", i, a[i], b[i], c[i]); }
        */
    }

    // free the resources on device side

    hipFree(dev_a);
    hipFree(dev_b);
    hipFree(dev_c);

    // free the resources on host side

    free(a);
    free(b);
    free(c);

    return errors;
}
