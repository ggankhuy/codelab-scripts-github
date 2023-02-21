/*Basic vector sum using hipMalloc */ 

#include <stdio.h>
#include "hip/hip_runtime.h"
#include <chrono>
#include <iostream>
#include <iomanip>

using namespace std;

__global__ void k1() {
    size_t start = clock64();
    size_t elapsed = 0;
    while (elapsed < 100000000) {
        elapsed = clock64() - start;
    }
}

#define N 64
#define N 1048576
#define ARRSIZE 3
#define LOOPSTRIDE 8
#define timer 0
#define HIPSETDEVICE 0
int main (void) {
    using namespace std::chrono;
    int *a, *b, *c;
    int *dev_a, *dev_b, *dev_c;
    int i ;

    char* env_timer;
    char* env_nocopy;

    string env_timer_str = "";
    string env_nocopy_str = "";

    env_timer=std::getenv("timer");
    env_nocopy=std::getenv("nocopy");

    env_timer ? env_timer_str=string(env_timer): "" ;
    env_nocopy ? env_nocopy_str=string(env_nocopy) : "";

    cout << "env_nocopy_str: " << env_nocopy_str << endl;

    if (env_nocopy_str != "1") {
        a = (int*)malloc(N * sizeof(int));
 	    hipMalloc(&dev_a, N * sizeof(int) );
    	for (int i = 0; i < N ; i ++ )
	    	a[i]  = i;
    } else {
        cout << "Bypassing hipMalloc/malloc..." << endl;
    } 

    high_resolution_clock::time_point t1, t2;

    if (env_timer_str != "1") {
        t1 = high_resolution_clock::now();
    }

    if (env_nocopy_str != "1") {
        printf("hipMemcpy.start.\n");
        hipMemcpy(dev_a, a, N * sizeof(int), hipMemcpyHostToDevice);
        printf("hipMemcpy.end\n");
    } else {
        cout << "Bypassing hipMemcpy..." << endl;
    }

    hipSetDevice(0);
	k1<<<1, 256, 0, 0>>>();

    #if HIPSETDEVICE == 1
    hipSetDevice(1);
	k1<<<1, 256, 0, 0>>>();
    hipSetDevice(2);
	k1<<<1, 256, 0, 0>>>();
    #endif

    if (env_timer_str != "1") {

        t2 = high_resolution_clock::now();
	    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
    	std::cout << "It took me " << time_span.count() << " seconds." << std::endl;
    }

    if (env_nocopy_str != "1") {
        hipFree(dev_a);
        free(a);
    } else {
        cout << "Bypassing hipFree/free..." << endl;
    }
	return 0;
}
