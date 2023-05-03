/*Basic vector sum using hipMalloc */ 

#include <stdio.h>
#include "hip/hip_runtime.h"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <string.h>
#include <stdlib.h> 
#include <ctime>
#include <ratio>

using namespace std;

// 512M

#define N 536870912
//#define N 8

int main (void) {
    using namespace std::chrono;
    int *dev_a, *dev_b, *dev_c;
    int i ;

    char* env_timer;

    string env_timer_str = "";
    env_timer=std::getenv("timer");
    env_timer ? env_timer_str=string(env_timer): "" ;

    high_resolution_clock::time_point t1, t2;

    if (env_timer_str != "1") {
        t1 = high_resolution_clock::now();
    }

    hipMalloc(&dev_a, N * sizeof(int) );

    if (env_timer_str != "1") {
        t2 = high_resolution_clock::now();
        duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
        std::cout << "It took me " << time_span.count() << " seconds." << std::endl;
    }

    hipFree(dev_a);

	return 0;
}
