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
#include <unistd.h>
#include "cmp_args.h"

#define SET_MAT_DIM(x,y,z) MAT_X=(x) ; MAT_Y=(y); MAT_Z=(z);
#define SET_TILE_DIM(x,y) T_X=(x) ; T_Y=(y); T_Z=1;
using namespace std;

// Device (Kernel) function, it must be void

/*
__global__ void matrixTranspose(int* out, int* in, const int width) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    out[y * width + x] = in[x * width + y];
}
*/

template<class T>
__global__ void add(T* a, T* b, T *c, const T nx, const T ny) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int tidx= y * nx + x;
    //if (x < nx && y < ny)
    c[tidx] = a[tidx] + b[tidx];
}

template <class T>
class matrix 
{
    public:

   
         T * a, * b, * c, *dev_a, *dev_b, *dev_c;
        int LOOPSTRIDE = 1;
        int MAT_X = 16;
        int MAT_Y = 16;
        int MAT_Z = 1;
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

            SET_MAT_DIM(X,Y,Z)
            SET_TILE_DIM(TILEX, TILEY)

            /*
            #if X==X8  MAT_X = 8; #endif
            #if X==X16  MAT_X = 16; #endif
            #if X==X64  MAT_X = 64; #endif
            #if X==X256  MAT_X = 256; #endif
            
            #if TILEY=1 T_Y=1 #endif
            #if TILEY=4 T_Y=4 #endif
            # ...
            */
            N=MAT_X*MAT_Y;

            /*
            if (env_project_name_str == "matrix_32x32_8x8x1") { MAT_X=32; MAT_Y=32; N=MAT_X*MAT_Y; T_X=8; T_Y=8; T_Z=1; }
            if (env_project_name_str == "matrix_32x32_4x4x1") { MAT_X=32; MAT_Y=32; N=MAT_X*MAT_Y; T_X=4; T_Y=4; T_Z=1;  }
            if (env_project_name_str == "matrix_256x256_32x32x1") { MAT_X=16; MAT_Y=64; N=MAT_X*MAT_Y; T_X=32; T_Y=32; T_Z=1; }
            if (env_project_name_str == "matrix_256x256_32x32x1_float") { MAT_X=16; MAT_Y=64; N=MAT_X*MAT_Y; T_X=32; T_Y=32; T_Z=1; }
            if (env_project_name_str == "matrix_256x256_64x16x1") { MAT_X=256; MAT_Y=256; N=MAT_X*MAT_Y; T_X=64; T_Y=16; T_Z=1; }
            if (env_project_name_str == "matrix_256x256_16x64x1") { MAT_X=256; MAT_Y=256; N=MAT_X*MAT_Y; T_X=16; T_Y=64; T_Z=1; }
            */
            LOOPSTRIDE=N/16;
            if (LOOPSTRIDE==0) {LOOPSTRIDE=1;}
            printf("Data initialized to MAT_X/Y=%d,%d, N=%d, T_X/Y/Z=%d, %d, %d.\n", MAT_X, MAT_Y, N, T_X, T_Y, T_Z);
            //sleep(3);
             
        }
        void initMatrix() {
            int acc = 0;

            for (int i = 0; i < N; i++) {
                a[i] = (T)i + acc;
                b[i] = (T)i * 4 + acc;

                if (i % MAT_X == 0) {
                    acc+=1024; 
                }
                c[i] = 999;
            }
        }

        void allocMem() {
                a = (T*)malloc(N * sizeof(T));  
                b = (T*)malloc(N * sizeof(T)); 
                c = (T*)malloc(N * sizeof(T));
                hipMalloc((void**)&dev_a, N * sizeof(T)); 
                hipMalloc((void**)&dev_b, N * sizeof(T));
                hipMalloc((void**)&dev_c, N * sizeof(T));
        }

        void dispResult(int pre_op=0) {
            for (int i = 0; i < N; i+=LOOPSTRIDE ) {
                #if DATATYPE==FP32
                    pre_op ? printf("Before add: %d: %f, %f.\n", i, a[i], b[i]) : printf("After add: %d: %f + %f = %f.\n", i, a[i], b[i], c[i]);
                #elif DATATYPE==INT32
                    pre_op ? printf("Before add: %d: %u, %u.\n", i, a[i], b[i]) : printf("After add: %d: %u + %u = %u.\n", i, a[i], b[i], c[i]);
                #else
                 #error "DATATYPE not specified p3."
                #endif
            }
        }

        void freeMem() {
            hipFree(dev_a);             
            hipFree(dev_b);
            hipFree(dev_c);
            free(a);
            free(b);
            free(c);
        }

        void memCpy(int h2d=1) {
            if (h2d == 1) {
                hipMemcpy(dev_a, a, N * sizeof(T), hipMemcpyHostToDevice);
                hipMemcpy(dev_b, b, N * sizeof(T), hipMemcpyHostToDevice);
                hipMemcpy(dev_c, c, N * sizeof(T), hipMemcpyHostToDevice);
            } else {
                hipMemcpy(a, dev_a, N * sizeof(T), hipMemcpyHostToDevice);
                hipMemcpy(b, dev_b, N * sizeof(T), hipMemcpyHostToDevice);
                hipMemcpy(c, dev_c, N * sizeof(T), hipMemcpyHostToDevice);
            }
        }    

        void memCpyH2D() { memCpy(1); }    
        void memCpyD2H() { memCpy(0); }    

        void callKernel() {
            #if DATATYPE==FP32
                printf("<<<dim3(%u, %u), (%u, %u)>>>, widthx/y: %u, %u.\n", MAT_X / T_X, MAT_Y / T_Y, T_X, T_Y, MAT_X, MAT_Y);
                add<float><<<dim3(MAT_X / T_X, MAT_Y / T_Y),  dim3(T_X, T_Y)>>>(dev_a, dev_b, dev_c, MAT_X, MAT_Y);
            #elif DATATYPE==INT32
                printf("<<<dim3(%u, %u), (%u, %u)>>>, widthx/y: %u, %u.\n", MAT_X / T_X, MAT_Y / T_Y, T_X, T_Y, MAT_X, MAT_Y);
                add<int><<<dim3(MAT_X / T_X, MAT_Y / T_Y),  dim3(T_X, T_Y)>>>(dev_a, dev_b, dev_c, MAT_X, MAT_Y);
            #else
             #error "DATATYPE not specified p7."
            #endif
        }
    private:
};

int main() {
    printf("Starting matrix computation...\n");
    #if DATATYPE==FP32
    matrix <float>m1;
    #elif DATATYPE==INT32
    matrix <int>m1;
    #else
     #error "DATA TYPE not specified (main)."
    #endif
    //m1.set_data();
    m1.allocMem();
    m1.initMatrix();
    m1.dispResult(1);
    m1.memCpyH2D();
    m1.callKernel();
    m1.memCpyD2H();
    m1.dispResult();
    m1.freeMem();
}
