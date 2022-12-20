#include <stdio.h>
#include <assert.h>
#include "hip/hip_runtime.h"

int main ( void ) {
    int device, i, count, stat;
    hipDeviceProp_t props;
    hipGetDeviceCount(&count);

    if (count > 10 || count < 0) {
        printf("Err: More than 10 devices or less than 0 devices!!!\n");
        return 1;
    } else {
        printf("Count: %d. \n", count);
    }

    
	for (i = 0 ; i < count ; i ++ ) {	
        int devId;
        stat = hipGetDevice(&devId);

        if (stat != hipSuccess)  {
            printf("Err: Unable to get device for device iD: %d.\n", i);
            continue;
        } else {
            printf("Devid found: %d.\n", devId);
        }

        hipSetDevice(device);
        hipGetDeviceProperties(&props, device);
        printf("info: running on device %s\n", props.name);

		hipGetDeviceProperties ( &props, i);
		printf("\n\n====DEVICE %d=====", i);
		printf("\nDevice name: 		%s", props.name);
		printf("\nTotal global mem: 	0x%0x", props.totalGlobalMem);
		printf("\nwarpSize: 		0x%0x", props.warpSize);
		printf("\nmaxThreads/Block: 	0x%0x", props.maxThreadsPerBlock);
		printf("\nmaxThreads/DIM: 	0x%0x", props.maxThreadsDim);
		printf("\nmaxGridSize: 		0x%0x", props.maxGridSize);
		printf("\ntotal const mem: 	0x%0x", props.totalConstMem);
		printf("\ncompute cap: 		%d.%d", props.major, props.minor);
		printf("\nmultiprocessors: 	0x%0x", props.multiProcessorCount);
		printf("\ncanMapHostMem: 	0x%0x", props.canMapHostMemory);
		printf("\ncomputeMode: 		0x%0x", props.computeMode);
		printf("\nconcurrentKernels: 	0x%0x", props.concurrentKernels);
	}

	printf("\n");
}
