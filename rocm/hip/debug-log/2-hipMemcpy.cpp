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

    char* env_timer;
    char* env_nocopy;

    try {
        env_timer = std::getenv("timer");
   } catch (const char* msg) {
        std::cerr << msg << std::endl;
   }

    try {
        env_nocopy = std::getenv("nocopy");
   } catch (const char* msg) {
        std::cerr << msg << std::endl;
   }


    a = (int*)malloc(N * sizeof(int));
 	hipMalloc(&dev_a, N * sizeof(int) );

	for (int i = 0; i < N ; i ++ )
		a[i]  = i;

    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    if (string(env_nocopy) == "0") {
        printf("hipMemcpy.start.\n");
        hipMemcpy(dev_a, a, N * sizeof(int), hipMemcpyHostToDevice);
        printf("hipMemcpy.end\n");
    }

    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
    std::cout << "It took me " << time_span.count() << " seconds." << std::endl;

    hipFree(dev_a);
    free(a);
    
	return 0;
}
