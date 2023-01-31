/*Basic vector sum using hipMalloc */ 

#include <stdio.h>
#include "hip/hip_runtime.h"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <string>

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

int main (void) {
    int *a, *b, *c;
    int *dev_a, *dev_b, *dev_c;
    int i ;

    string env_timer = getenv("timer");
    cout << "env_timer: " << env_timer << endl;

    string env_no_copy = getenv("no_copy");
    cout << "env_no_copy: " << env_no_copy << endl;

    auto end = std::chrono::high_resolution_clock::now();
    auto start = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::high_resolution_clock::now();
    auto ns = std::chrono::high_resolution_clock::now();
    int ns_fractional;
    a = (int*)malloc(N * sizeof(int));
 	hipMalloc(&dev_a, N * sizeof(int) );

	for (int i = 0; i < N ; i ++ )
		a[i]  = i;

    //if (env_timer == "1")
    start = std::chrono::high_resolution_clock::now();

    //if (env_no_copy == "1") {
        printf("hipMemcpy.start.\n");
   	    hipMemcpy(dev_a, a, N * sizeof(int), hipMemcpyHostToDevice);
        printf("hipMemcpy.end\n");
    }

    //if (env_timer == "1") {
    end = std::chrono::high_resolution_clock::now();
    duration = (end - start);
    ns = std::chrono::duration_cast<std::chrono::nanoseconds>(duration);
    ns_fractional = static_cast<int>(ns.count());
    cout << setw(30) << "hipMemcpy duration: " << ns_fractional << " ns or " << ns_fractional / 1000000 << " ms" << endl;
    

    hipFree(dev_a);
    free(a);
    
	return 0;
}
