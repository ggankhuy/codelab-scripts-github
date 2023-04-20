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
//#include "vector.h"
#include <iostream>
#include <string.h>

#define ARG_OP_ADD 1
#define ARG_OP_MUL 2
#define ARG_OP_MUL_ADD 3

#define ARG_DATATYPE_INT32 1
#define ARG_DATATYPE_FP32 2

#define ARG_DATASIZE_VEC_4 1
#define ARG_DATASIZE_VEC_8 2
#define ARG_DATASIZE_VEC_64 3
#define ARG_DATASIZE_VEC_1024 4

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

__global__ void addFloat(float* a, float* b, float *c, const int nx, const int ny) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int tidx= y * nx + x;
    //if (x < nx && y < ny)
    c[tidx] = a[tidx] + b[tidx];
}

class matrix 
{
    public:

   
        #if DATATYPE==ARG_DATATYPE_INT32
         int * a, * b, * c, *dev_a, *dev_b, *dev_c;
        #elif DATATYPE==ARG_DATATYPE_FP32
         float *a, *b, *c,  *dev_a, *dev_b, *dev_c;
        #else
         #error "DATATYPE not specified."
        #endif
        int LOOPSTRIDE = 1;
        int MAT_X = 16;
        int MAT_Y = 16;
        int N = (MAT_X * MAT_Y);
        int T_X = 4;
        int T_Y = 4;
        int T_Z = 1;
        char* env_project_name;
        string env_project_name_str = "";

        matrix() {
            this->set_data();
        }

        void set_data() {
            env_project_name=std::getenv("PROJECT_NAME");
            env_project_name ? env_project_name_str=string(env_project_name): "" ;

            if (env_project_name_str == "matrix_32x32_8x8x1") { MAT_X=32; MAT_Y=32; N=MAT_X*MAT_Y; T_X=8; T_Y=8; T_Z=1; }
            if (env_project_name_str == "matrix_32x32_4x4x1") { MAT_X=32; MAT_Y=32; N=MAT_X*MAT_Y; T_X=4; T_Y=4; T_Z=1;  }
            if (env_project_name_str == "matrix_256x256_32x32x1") { MAT_X=16; MAT_Y=64; N=MAT_X*MAT_Y; T_X=32; T_Y=32; T_Z=1; }
            if (env_project_name_str == "matrix_256x256_32x32x1_float") { MAT_X=16; MAT_Y=64; N=MAT_X*MAT_Y; T_X=32; T_Y=32; T_Z=1; }
            if (env_project_name_str == "matrix_256x256_64x16x1") { MAT_X=256; MAT_Y=256; N=MAT_X*MAT_Y; T_X=64; T_Y=16; T_Z=1; }
            if (env_project_name_str == "matrix_256x256_16x64x1") { MAT_X=256; MAT_Y=256; N=MAT_X*MAT_Y; T_X=16; T_Y=64; T_Z=1; }
            LOOPSTRIDE=N/16;
        }
        void initMatrix() {
            int acc = 0;

            for (int i = 0; i < N; i++) {
                #if DATATYPE==ARG_DATATYPE_INT32
                    a[i] = (int)i + acc;
                    b[i] = (int)i * 4 + acc;
                #elif DATATYPE==ARG_DATATYPE_FP32
                    a[i] = (float)i + acc;
                    b[i] = (float)i * 4 + acc;
                #else
                 #error "DATATYPE not specified."
                #endif

                if (i % MAT_X == 0) {
                    acc+=1024; 
                }
                c[i] = 999;
            }
        }

        void allocMem() {
            #if DATATYPE==ARG_DATATYPE_FP32
                a = (float*)malloc(N * sizeof(float));  
                b = (float*)malloc(N * sizeof(float)); 
                c = (float*)malloc(N * sizeof(float));
                hipMalloc((void**)&dev_a, N * sizeof(float)); 
                hipMalloc((void**)&dev_b, N * sizeof(float));
                hipMalloc((void**)&dev_c, N * sizeof(float));
            #elif DATATYPE==ARG_DATATYPE_INT32
                a = (int*)malloc(N * sizeof(int));  
                b = (int*)malloc(N * sizeof(int)); 
                c = (int*)malloc(N * sizeof(int));
                hipMalloc((void**)&dev_a, N * sizeof(int)); 
                hipMalloc((void**)&dev_b, N * sizeof(int));
                hipMalloc((void**)&dev_c, N * sizeof(int));
            #else
             #error "DATATYPE not specified p2."
            #endif
        }

        void dispResult(int pre_op=0) {
            for (int i = 0; i < N; i+=LOOPSTRIDE ) {
                #if DATATYPE==ARG_DATATYPE_FP32
                    pre_op ? printf("Before add: %d: %f, %f.\n", i, a[i], b[i]) : printf("After add: %d: %f + %f = %f.\n", i, a[i], b[i], c[i]);
                #elif DATATYPE==ARG_DATATYPE_INT32
                    pre_op ? printf("Before add: %d: %u, %u.\n", i, a[i], b[i]) : printf("After add: %d: %u + %u = %u.\n", i, a[i], b[i], c[i]);
                #else
                 #error "DATATYPE not specified p3."
                #endif
            }
        }

        void freeMem() {
            #if DATATYPE==ARG_DATATYPE_FP32
            hipFree(dev_a);             
            hipFree(dev_b);
            hipFree(dev_c);
            free(a);
            free(b);
            free(c);
            #elif DATATYPE==ARG_DATATYPE_INT32
            hipFree(dev_a);
            hipFree(dev_b);
            hipFree(dev_c);
            free(a);
            free(b);
            free(c);
            #else
             #error "DATATYPE not specified p4."
            #endif
        }

        void memCpyD2H() {
            #if DATATYPE==ARG_DATATYPE_FP32
                hipMemcpy(dev_a, a, N * sizeof(float), hipMemcpyHostToDevice);
                hipMemcpy(dev_b, b, N * sizeof(float), hipMemcpyHostToDevice);
                hipMemcpy(dev_c, c, N * sizeof(float), hipMemcpyHostToDevice);
            #elif DATATYPE==ARG_DATATYPE_INT32
                hipMemcpy(dev_a, a, N * sizeof(int), hipMemcpyHostToDevice);
                hipMemcpy(dev_b, b, N * sizeof(int), hipMemcpyHostToDevice);
                hipMemcpy(dev_c, c, N * sizeof(int), hipMemcpyHostToDevice);
            #else
             #error "DATATYPE not specified p5."
            #endif
        }    

        void memCpyH2D() {
            #if DATATYPE==ARG_DATATYPE_FP32
                hipMemcpy(a, dev_a, N * sizeof(float), hipMemcpyHostToDevice);
                hipMemcpy(b, dev_b, N * sizeof(float), hipMemcpyHostToDevice);
                hipMemcpy(c, dev_c, N * sizeof(float), hipMemcpyHostToDevice);
            #elif DATATYPE==ARG_DATATYPE_INT32
                hipMemcpy(dev_a, a, N * sizeof(int), hipMemcpyHostToDevice);
                hipMemcpy(dev_b, b, N * sizeof(int), hipMemcpyHostToDevice);
                hipMemcpy(dev_c, c, N * sizeof(int), hipMemcpyHostToDevice);
            #else
             #error "DATATYPE not specified p6."
            #endif
        }    

        void callKernel() {
            #if DATATYPE==ARG_DATATYPE_FP32
                printf("<<<dim3(%u, %u), (%u, %u)>>>, widthx/y: %u, %u.\n", MAT_X / T_X, MAT_Y / T_Y, T_X, T_Y, MAT_X, MAT_Y);
                addFloat<<<dim3(MAT_X / T_X, MAT_Y / T_Y),  dim3(T_X, T_Y)>>>(dev_a, dev_b, dev_c, MAT_X, MAT_Y);
            #elif DATATYPE==ARG_DATATYPE_INT32
                printf("<<<dim3(%u, %u), (%u, %u)>>>, widthx/y: %u, %u.\n", MAT_X / T_X, MAT_Y / T_Y, T_X, T_Y, MAT_X, MAT_Y);
                add<<<dim3(MAT_X / T_X, MAT_Y / T_Y),  dim3(T_X, T_Y)>>>(dev_a, dev_b, dev_c, MAT_X, MAT_Y);
            #else
             #error "DATATYPE not specified p7."
            #endif
        }
    private:
};

int main() {
    printf("Starting matrix computation...\n");
    matrix m1;
    m1.set_data();
    m1.allocMem();
    m1.initMatrix();
    m1.dispResult(1);
    m1.memCpyH2D();
    m1.callKernel();
    m1.memCpyD2H();
    m1.dispResult();
    m1.freeMem();
}
