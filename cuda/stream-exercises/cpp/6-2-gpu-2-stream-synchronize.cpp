/*Basic vector sum using hipMalloc */ 

#include <stdio.h>
#include <cstdlib>
#include "hip/hip_runtime.h"

#define STREAMS 3
#define N 1024
#define SYNC_TYPE_NO_SYNC 0
#define SYNC_TYPE_PER_DEVICE_SYNC 1
#define SYNC_TYPE_PER_HOST_SYNC 2
//#define CONFIG_SYNC_OPTION SYNC_TYPE_NO_SYNC
//#define CONFIG_SYNC_OPTION SYNC_TYPE_PER_DEVICE_SYNC
#define CONFIG_SYNC_OPTION SYNC_TYPE_PER_HOST_SYNC

#define STREAM_PLACE_UNDEFINED 0 
#define STREAM_PLACE_OPT_DEVICE_1 1
#define STREAM_PLACE_OPT_PER_DEVICE 2
//#define CONFIG_STREAM_PLACEMENT STREAM_PLACE_OPT_DEVICE_1
#define CONFIG_STREAM_PLACEMENT STREAM_PLACE_OPT_PER_DEVICE

__global__ void k1(int *a) {
    size_t start = clock64();
    size_t elapsed = 0;
    while (elapsed < 100000000) {
        elapsed = clock64() - start;
    }
}

__global__ void k2(int *a) {
    size_t start = clock64();
    size_t elapsed = 0;
    while (elapsed < 300000000) {
        elapsed = clock64() - start;
    }
}

__global__ void k3(int *a) {
    size_t start = clock64();
    size_t elapsed = 0;
    while (elapsed < 300000000) {
        elapsed = clock64() - start;
    }
}

int main (void) {
    int *a, *b, *c;
    int *dev_a, *dev_b, *dev_c;
    int i, ret;
    //unsigned int CONFIG_SYNC_OPTION = SYNC_TYPE_NO_SYNC;

    hipStream_t streams[STREAMS];

    for (i = 0 ; i < STREAMS; i++) {
        switch(CONFIG_STREAM_PLACEMENT) {
            case STREAM_PLACE_OPT_DEVICE_1:
                printf("Placing all streams on 1st device only explicitly.\n");
                ret = hipSetDevice(0);
                break;
            case STREAM_PLACE_OPT_PER_DEVICE:
                printf("Placing all streams on each device .\n");
                ret = hipSetDevice(i);
                break;
            default:
                printf("No explicit stream placement, likely on first device by default.\n");
        }

        if (ret!= 0) {
            printf("hipSetDevice() for device %u failed.\n", i);
            return 1;
        } 
        
        ret = hipStreamCreate(&streams[i]);
        if (ret != 0) {
            printf("Stream %u create failed.\n", i);
            return 1;
        }
    }
    
    a = (int*)malloc(N * sizeof(int));
    hipMalloc(&dev_a, N * sizeof(int) );

    for (int i = 0; i < N ; i ++ ) {
        a[i]  = i;
    }

    const unsigned blocks = 256;
    const unsigned threadsPerBlock = 1;

    hipMemcpyAsync(dev_a, a, N * sizeof(int), hipMemcpyHostToDevice, streams[0]);

    printf("Launching streams...\n");
    k1<<<256,1,0,streams[0]>>>(dev_a);
    k2<<<256,1,0,streams[1]>>>(dev_a);

    printf("Curr sync option: %u.\n", CONFIG_SYNC_OPTION);
    switch(CONFIG_SYNC_OPTION) {
        case SYNC_TYPE_NO_SYNC:
            printf("No synchronization enabled.\n");
            break;
        case SYNC_TYPE_PER_DEVICE_SYNC:
            printf("Per device synchronization only.\n");
            hipDeviceSynchronize();
            break;
        case SYNC_TYPE_PER_HOST_SYNC:
            printf("Per host synchronization all GPUs on current host.\n");

            for (i = 0 ; i < STREAMS ; i++ ) {
                hipStreamSynchronize(streams[i]);
            }
            break;
        default:
            printf("Unsupport sync type %u.\n", CONFIG_SYNC_OPTION);
    }
    hipSetDevice(0);
    k3<<<256,1,0,streams[2]>>>(dev_a);

    hipMemcpyAsync(a, dev_a, N * sizeof(int), hipMemcpyDeviceToHost, streams[0]);

    hipFree(dev_a);
    free(a);

	return 0;
}
