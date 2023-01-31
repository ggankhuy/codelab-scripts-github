/*Basic vector sum using hipMalloc */ 

#include <stdio.h>
#include "hip/hip_runtime.h"
#include <chrono>
#include <iostream>
#include <iomanip>

using namespace std;

// 512M

#define N 536870912
//#define N 8
#define ARRSIZE 3
#define LOOPSTRIDE 8
__global__ void add(int *a, int*b, int *c) {
	int tid = hipBlockIdx_x;
	c[tid] = a[tid] + b[tid];
}

#define timer 0

int main (void) {
    int *a, *b, *c;
    int *dev_a, *dev_b, *dev_c;
    int i ;

    #if timer == 1
    auto end = std::chrono::high_resolution_clock::now();
    auto start = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::high_resolution_clock::now();
    auto ns = std::chrono::high_resolution_clock::now();
    #endif 

    a = (int*)malloc(N * sizeof(int));

 	hipMalloc(&dev_a, N * sizeof(int) );

	for (int i = 0; i < N ; i ++ ) {
		a[i]  = i;
	}

    #if timer == 1
    auto start = std::chrono::high_resolution_clock::now();
    #endif

    /*printf("hipMemcpy.start.\n");
   	hipMemcpy(dev_a, a, N * sizeof(int), hipMemcpyHostToDevice);
    printf("hipMemcpy.end\n");*/

    #if timer == 1
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = (end - start);
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(duration);
    int ns_fractional = static_cast<int>(ns.count());
    cout << setw(30) << "hipMemcpy duration: " << ns_fractional << " ns or " << ns_fractional / 1000000 << " ms" << endl;
    #endif

    hipFree(dev_a);
    free(a);
    
	return 0;
}
