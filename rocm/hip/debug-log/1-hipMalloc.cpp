/*Basic vector sum using hipMalloc */ 

#include <stdio.h>
#include "hip/hip_runtime.h"
#include <chrono>
#include <iostream>
#include <iomanip>

using namespace std;

#define N 64
#define N 1048576
#define ARRSIZE 3
#define LOOPSTRIDE 8
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

    int ns_fractional;    

    for (int i = 0; i < 3; i++ ) {
        #if timer == 1
        start = std::chrono::high_resolution_clock::now();
        #endif

     	hipMalloc(&dev_a, N * sizeof(int) );
    
        #if timer == 1
        end = std::chrono::high_resolution_clock::now();
        duration = (end - start);
        ns = std::chrono::duration_cast<std::chrono::nanoseconds(duration);
        ns_fractional = static_cast<int>(ns.count());
        cout << setw(30) << "hipMalloc duration: " << ns_fractional << " ns or " << ns_fractional / 1000000 << " ms" << endl;
        #endif

        #if timer == 1
        start = std::chrono::high_resolution_clock::now();
        #endif

        hipFree(dev_a);

        end = std::chrono::high_resolution_clock::now();
        duration = (end - start);
        ns = std::chrono::duration_cast<std::chrono::nanoseconds>(duration);
        ns_fractional = static_cast<int>(ns.count());
        cout << setw(30) << "hipFree duration: " << ns_fractional << " ns or " << ns_fractional / 1000000 << " ms " << endl;
    }
	return 0;
}
