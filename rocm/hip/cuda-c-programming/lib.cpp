#include <stdio.h>
#include <sys/time.h>
#include <stdbool.h>
#include <lib.h>

double seconds() {
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
            printf("different on %dth element: host %f gpu %f\n", i, hostRef[i], gpuRef[i]);
            break;

        }
    }
    if (!match)  printf("Arrays do not match.\n\n");
}

void sumMatrixOnHost(float *A, float *B, float *C, const int nx, const int ny) { 
        float *ia = A;
        float *ib = B;
        float *ic = C;

        for (int iy=0; iy<ny; iy++) { 
                ia += nx; ib += nx, ic += nx;
        }
}

void sumArraysOnHost(float * A, float *B, float *C, const int N) {
    for (int idx = 0; idx < N ; idx++) 
        C[idx] = A[idx] + B[idx];
}

void sumArraysOnHost(float *A, float *B, float *C, const int n, int offset)
{
    for (int idx = offset, k = 0; idx < n; idx++, k++)
    {
        C[k] = A[idx] + B[idx];
    }
}

void printData(char *msg, int *in,  const int size)
{
    printf("%s: ", msg);

    for (int i = 0; i < size; i++)
    {
        printf("%5d", in[i]);
        fflush(stdout);
    }

    printf("\n");
    return;
}
