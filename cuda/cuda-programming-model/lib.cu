#include <stdio.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <stdbool.h>
#include <lib.h>

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

void initialData(float * ip, int size) {

    // generate different seed for random number.

    time_t t;
    srand((unsigned) time (&t));

    for (int i = 0; i < size; i ++ ) {
        ip[i] = (float)(rand() & 0xFF ) / 10.0f;
    }
}

void checkResult(float * hostRef, float * gpuRef, const int N) {
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < N; i++) 
    {
        if (abs(hostRef[i] - gpuRef[i] > epsilon)) 
        {
            match = 0;
            printf("Arrays do not match!\n");
            printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i], gpuRef[i], i);
            break;

        }
    }
}

void sumMatrixOnHost(float *A, float *B, float *C, const int nx, const int ny) { 
        float *ia = A;
        float *ib = B;
        float *ic = C;

        for (int iy=0; iy<ny; iy++) { 
                ia += nx; ib += nx, ic += nx;
        }
}
