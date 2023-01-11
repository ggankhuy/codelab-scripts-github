/*Basic vector sum using hipMalloc */ 

#include <stdio.h>
#include "hip/hip_runtime.h"
#include <chrono>
#include <iostream>
#include <iomanip>

using namespace std;

#define N 64
#define ARRSIZE 3
#define LOOPSTRIDE 8

int main (void) {
    int *a, *b, *c;
    int *dev_a, *dev_b, *dev_c;
    int i ;
    
    #if timer == 1
    auto start = std::chrono::high_resolution_clock::now();
    #endif

 	hipMalloc(&dev_a, N * sizeof(int) );

    #if timer == 1
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = (end - start);
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(duration);
    int ns_fractional = static_cast<int>(ns.count());
    cout << setw(30) << "hipMalloc duration: " << ns_fractional << " ns or " << ns_fractional / 1000000 << " ms" << endl;
    #endif

    #if timer == 1
    start = std::chrono::high_resolution_clock::now();
    #endif

    hipFree(dev_a);

    #if timer == 1
    end = std::chrono::high_resolution_clock::now();
    duration = (end - start);
    ns = std::chrono::duration_cast<std::chrono::nanoseconds>(duration);
    ns_fractional = static_cast<int>(ns.count());
    cout << setw(30) << "hipFree duration: " << ns_fractional << " ns or " << ns_fractional / 1000000 << " ms " << endl;
    #endif

	return 0;
}
