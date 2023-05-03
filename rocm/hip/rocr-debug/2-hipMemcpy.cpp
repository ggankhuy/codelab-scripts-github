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
#include <ctype.h>

using namespace std;

// 512M

#define N 536870912
//#define N 8
#define ARRSIZE 3
#define LOOPSTRIDE 8

int main (void) {
    using namespace std::chrono;
    int *a, *b, *c;
    int *dev_a, *dev_b, *dev_c;
    int i ;
    int datasize_bytes;

    char* env_timer;
    char* env_nocopy;
    char* env_datasize;

    string env_timer_str = "";
    string env_nocopy_str = "";
    string env_datasize_str = "";

    env_timer=std::getenv("timer");
    env_nocopy=std::getenv("nocopy");
    env_datasize=std::getenv("datasize");

    env_timer ? env_timer_str=string(env_timer): "" ;
    env_nocopy ? env_nocopy_str=string(env_nocopy) : "";
    env_datasize ? env_datasize_str=string(env_datasize) : "";

    cout << "env_nocopy_str: " << env_nocopy_str << endl;

    if (env_datasize_str != "") {
        cout << "Setting N to " << env_datasize_str <<  "MB..." << endl;
        datasize_bytes = stoi(env_datasize_str) * 1024 * 1024;
    } else {
        cout << "Error: env_datasize_str is not integer: " << env_datasize_str << " Leaving the default size of " << N / 1024/1024 << " MB." << endl;
        datasize_bytes = N;
    }

    if (env_nocopy_str != "1") {
        a = (int*)malloc(datasize_bytes * sizeof(int));
 	    hipMalloc(&dev_a, datasize_bytes * sizeof(int) );
    	for (int i = 0; i < datasize_bytes ; i ++ )
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
        hipMemcpy(dev_a, a, datasize_bytes * sizeof(int), hipMemcpyHostToDevice);
        printf("hipMemcpy.end\n");
    } else {
        cout << "Bypassing hipMemcpy..." << endl;
    }

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
