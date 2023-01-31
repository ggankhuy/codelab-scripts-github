/*Basic vector sum using hipMalloc */ 

#include <stdio.h>
#include "hip/hip_runtime.h"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <string>

#include <ctime>
#include <ratio>

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
    using namespace std::chrono;
    int *a, *b, *c;
    int *dev_a, *dev_b, *dev_c;
    int i ;

    string env_timer = getenv("timer");
    cout << "env_timer: " << env_timer << endl;

    string env_no_copy = getenv("no_copy");
    cout << "env_no_copy: " << env_no_copy << endl;

    a = (int*)malloc(N * sizeof(int));
 	hipMalloc(&dev_a, N * sizeof(int) );

	for (int i = 0; i < N ; i ++ )
		a[i]  = i;

    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    if (env_no_copy == "1") {
        printf("hipMemcpy.start.\n");
   	    hipMemcpy(dev_a, a, N * sizeof(int), hipMemcpyHostToDevice);
        printf("hipMemcpy.end\n");
    }

    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    hipFree(dev_a);
    free(a);
    
	return 0;
}
