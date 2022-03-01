/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */

#include <stdio.h>
#include <stdbool.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define SIZE    (64*1024*1024)

float cuda_malloc_test( int size, bool up ) {
    cudaEvent_t     start, stop;
    int             *a, *dev_a;
    float           elapsedTime;

     cudaEventCreate( &start );
     cudaEventCreate( &stop );

    a = (int*)malloc( size * sizeof( *a ) );
     cudaMalloc( (void**)&dev_a,
                              size * sizeof( *dev_a )) ;

     cudaEventRecord( start, 0 );
    for (int i=0; i<100; i++) {
        if (up)
             cudaMemcpy( dev_a, a,
                                  size * sizeof( *dev_a ),
                                  cudaMemcpyHostToDevice );
        else
             cudaMemcpy( a, dev_a,
                                  size * sizeof( *dev_a ),
                                  cudaMemcpyDeviceToHost );
    }
     cudaEventRecord( stop, 0 );
     cudaEventSynchronize( stop );
     cudaEventElapsedTime( &elapsedTime,
                                        start, stop );

    free( a );
     cudaFree( dev_a );
     cudaEventDestroy( start );
     cudaEventDestroy( stop );

    return elapsedTime;
}


float cuda_host_alloc_test( int size, bool up ) {
    cudaEvent_t     start, stop;
    int             *a, *dev_a;
    float           elapsedTime;

     cudaEventCreate( &start );
     cudaEventCreate( &stop );

     cudaHostAlloc( (void**)&a,
                                 size * sizeof( *a ),
                                 cudaHostAllocDefault );
     cudaMalloc( (void**)&dev_a,
                              size * sizeof( *dev_a ) );

     cudaEventRecord( start, 0 );
    for (int i=0; i<100; i++) {
        if (up)
             cudaMemcpy( dev_a, a,
                                  size * sizeof( *a ),
                                  cudaMemcpyHostToDevice );
        else
             cudaMemcpy( a, dev_a,
                                  size * sizeof( *a ),
                                  cudaMemcpyDeviceToHost );
    }
     cudaEventRecord( stop, 0 );
     cudaEventSynchronize( stop );
     cudaEventElapsedTime( &elapsedTime,
                                        start, stop );

     cudaFreeHost( a );
     cudaFree( dev_a );
     cudaEventDestroy( start );
     cudaEventDestroy( stop );

    return elapsedTime;
}


int main( void ) {
    float           elapsedTime;
    float           MB = (float)100*SIZE*sizeof(int)/1024/1024;


    // try it with cudaMalloc
    elapsedTime = cuda_malloc_test( SIZE, true );
    printf( "Time using cudaMalloc:  %3.1f ms\n",
            elapsedTime );
    printf( "\tMB/s during copy up:  %3.1f\n",
            MB/(elapsedTime/1000) );

    elapsedTime = cuda_malloc_test( SIZE, false );
    printf( "Time using cudaMalloc:  %3.1f ms\n",
            elapsedTime );
    printf( "\tMB/s during copy down:  %3.1f\n",
            MB/(elapsedTime/1000) );

    // now try it with cudaHostAlloc
    elapsedTime = cuda_host_alloc_test( SIZE, true );
    printf( "Time using cudaHostAlloc:  %3.1f ms\n",
            elapsedTime );
    printf( "\tMB/s during copy up:  %3.1f\n",
            MB/(elapsedTime/1000) );

    elapsedTime = cuda_host_alloc_test( SIZE, false );
    printf( "Time using cudaHostAlloc:  %3.1f ms\n",
            elapsedTime );
    printf( "\tMB/s during copy down:  %3.1f\n",
            MB/(elapsedTime/1000) );
}
