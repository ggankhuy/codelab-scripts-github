/*Basic vector sum using hipMalloc */ 

#include <stdio.h>
#include "hip/hip_runtime.h"
//#include "vector.h"
#include <iostream>
#include <string.h>

#define ARRSIZE 3

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

__global__ void add(int *a, int*b, int *c) {
	int tid = hipBlockIdx_x;
	c[tid] = a[tid] + b[tid];
}

__global__ void mul(int *a, int*b, int *c) {
	int tid = hipBlockIdx_x;
	c[tid] = a[tid] * b[tid];
}

__global__ void mul_add(int *a, int*b, int *c) {
	int tid = hipBlockIdx_x;
	c[tid] = a[tid] * 2 + b[tid];
}

int main (void) {
    int *a, *b, *c;
    int *dev_a, *dev_b, *dev_c;
    int i ;

    int N = 64;
    int LOOPSTRIDE = 8;

    // process project_name env var.

    char* env_project_name;
    string env_project_name_str = "";
    env_project_name=std::getenv("PROJECT_NAME");
    env_project_name ? env_project_name_str=string(env_project_name): "" ;

    if (env_project_name_str == "vector4") { N = 4; LOOPSTRIDE=1; }
    if (env_project_name_str == "vector1024") { N = 1024; LOOPSTRIDE=N/16; }
    printf("N/LOOPSTRIDE: %u, %u.\n", N, LOOPSTRIDE);

    a = (int*)malloc(N * sizeof(int));
    b = (int*)malloc(N * sizeof(int));
    c = (int*)malloc(N * sizeof(int));
 	hipMalloc(&dev_a, N * sizeof(int) );
 	hipMalloc(&dev_b, N * sizeof(int) );
 	hipMalloc(&dev_c, N * sizeof(int) );

	for (int i = 0; i < N ; i ++ ) {
		a[i]  = i;
		b[i] = i + i;
		c[i] = 999;
	}

	for (int i = 0; i < N ; i+=LOOPSTRIDE ) {
        #if OP==ARG_OP_ADD
         printf("Before add: a/b: %d, %d.\n", a[i], b[i]);
        #elif OP==ARG_OP_MUL
         printf("Before mul: a/b: %d, %d.\n", a[i], b[i]);
        #elif OP==ARG_OP_MUL_ADD
         printf("Before mul + add: a/b: %d, %d.\n", a[i], b[i]);
        #else
         #error "OP not specified. Can not compiler further."
        #endif
	}

   	hipMemcpy(dev_a, a, N * sizeof(int), hipMemcpyHostToDevice);
   	hipMemcpy(dev_b, b, N * sizeof(int), hipMemcpyHostToDevice);
   	hipMemcpy(dev_c, c, N * sizeof(int), hipMemcpyHostToDevice);
    
    const unsigned blocks = 256;
    const unsigned threadsPerBlock = 1;

    #if OP==ARG_OP_ADD
     add<<<blocks, threadsPerBlock>>>(dev_a, dev_b, dev_c);
    #elif OP==ARG_OP_MUL
     mul<<<blocks, threadsPerBlock>>>(dev_a, dev_b, dev_c);
    #elif OP==ARG_OP_MUL_ADD
     mul_add<<<blocks, threadsPerBlock>>>(dev_a, dev_b, dev_c);
    #else
     #error "OP not specified. Can not compiler further."
    #endif

    hipMemcpy(a, dev_a, N * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(b, dev_b, N * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(c, dev_c, N * sizeof(int), hipMemcpyDeviceToHost);

	for (int i = 0; i < N; i+=LOOPSTRIDE ) {
        #if OP==ARG_OP_ADD
         printf("After add: %d: %u + %u = %u\n", i, a[i], b[i], c[i]);
        #elif OP==ARG_OP_MUL
         printf("After mul: %d: %u + %u = %u\n", i, a[i], b[i], c[i]);
        #elif OP==ARG_OP_MUL_ADD
         printf("After uml_add: %d: %u + %u = %u\n", i, a[i], b[i], c[i]);
        #else
         #error "OP not specified. Can not compiler further."
        #endif
    }

    hipFree(dev_a);
    hipFree(dev_b);
    hipFree(dev_c);
    free(a);
    free(b);
    free(c);
    
	return 0;
}
