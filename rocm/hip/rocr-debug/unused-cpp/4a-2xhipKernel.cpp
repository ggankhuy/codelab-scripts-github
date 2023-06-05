
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

__global__ void k2() {
    size_t start = clock64();
    size_t elapsed = 0;
    while (elapsed < 200000000) {
        elapsed = clock64() - start;
    }
}

#define N 64
#define N 1048576
#define ARRSIZE 3
#define LOOPSTRIDE 8
#define timer 0
int main (void) {

    #if timer == 1
    auto end = std::chrono::high_resolution_clock::now();
    auto start = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::high_resolution_clock::now();
    auto ns = std::chrono::high_resolution_clock::now();
    #endif 

    int ns_fractional;    

    for (int i = 0; i < 3; i++ ) {
        #if timer == 1
        start = std::chrono::high_resolution_clock::now();
        #endif

        k1<<<1, 256, 0, 0>>>();
        k2<<<1, 256, 0, 0>>>();

        #if timer == 1
        end = std::chrono::high_resolution_clock::now();
        duration = (end - start);
        ns = std::chrono::duration_cast<std::chrono::nanoseconds(duration);
        ns_fractional = static_cast<int>(ns.count());
        cout << setw(30) << "hipMalloc duration: " << ns_fractional << " ns or " << ns_fractional / 1000000 << " ms" << endl;
        #endif
    }
	return 0;
}
