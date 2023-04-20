/*Basic vector sum using hipMalloc */ 

#include <stdio.h>
#include "hip/hip_runtime.h"
//#include "vector.h"
#include <iostream>
#include <string.h>

/*
#define N 64
#define LOOPSTRIDE 8
#ifdef OPT_SUB_PROJECT_NAME
 #undef LOOPSTRIDE
 #undef N

 #if OPT_SUB_PROJECT_NAME == vector4
  #define N 4
  #define LOOPSTRIDE 1
 #elif OPT_SUB_PROJECT_NAME == vector1024
  #define N 1024
  #define LOOPSTRIDE 32
 #endif
#endif
*/

#define ARRSIZE 3

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

class matrix 
{
    public:
        int * a, * b, * c, *dev_a, *dev_b, *dev_c;
        float *f32_a, *f32_b, *f32_c, *f32_dev_a, *f32_dev_b, *f32_dev_c;

        void memCpyD2H() {
        }    

        void memCpyH2D() {
        }    

        void callKernel() {
        }
    private:
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

    // process OP env var.
    char* env_op;
    string env_op_str = "";
    env_op=std::getenv("OP");
    env_op ? env_op_str=string(env_op): "" ;

    cout << "env_op: string: " << env_op_str << endl;
    
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
        if (env_op_str == "add") { printf("Before add: a/b: %d, %d.\n", a[i], b[i]); }
        if (env_op_str == "mul") { printf("Before mul: a/b: %d, %d.\n", a[i], b[i]); }
        if (env_op_str == "mul_add") { printf("Before mul_add: a/b: %d, %d.\n", a[i], b[i]); }
	}

   	hipMemcpy(dev_a, a, N * sizeof(int), hipMemcpyHostToDevice);
   	hipMemcpy(dev_b, b, N * sizeof(int), hipMemcpyHostToDevice);
   	hipMemcpy(dev_c, c, N * sizeof(int), hipMemcpyHostToDevice);
    
    const unsigned blocks = 256;
    const unsigned threadsPerBlock = 1;

    if (env_op_str == "add") { cout << "Launching add()..." << endl; hipLaunchKernelGGL(add, blocks, threadsPerBlock, 0, 0, dev_a, dev_b, dev_c); }
    if (env_op_str == "mul") { cout << "Launching mul()..." << endl; hipLaunchKernelGGL(mul, blocks, threadsPerBlock, 0, 0, dev_a, dev_b, dev_c); }
    if (env_op_str == "mul_add") { cout << "Launching mul_add()" << endl; hipLaunchKernelGGL(mul_add, blocks, threadsPerBlock, 0, 0, dev_a, dev_b, dev_c); }

    hipMemcpy(a, dev_a, N * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(b, dev_b, N * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(c, dev_c, N * sizeof(int), hipMemcpyDeviceToHost);

	for (int i = 0; i < N; i+=LOOPSTRIDE ) {
        if (env_op_str == "add") { printf("After add: %d: %u + %u = %u\n", i, a[i], b[i], c[i]); }
        if (env_op_str == "mul") { printf("After mul: %d: %u * %u = %u\n", i, a[i], b[i], c[i]); }
        if (env_op_str == "mul_add") { printf("After mul_add: %d: %u * 2 + %u = %u\n", i, a[i], b[i], c[i]); }
    }

    hipFree(dev_a);
    hipFree(dev_b);
    hipFree(dev_c);
    free(a);
    free(b);
    free(c);
    
	return 0;
}
