#include <stdio.h>
#include <sys/time.h>
#include <stdbool.h>

double seconds();
void initialData(float * ip, int size);
void checkResult(float * hostRef, float * gpuRef, const int N);
void sumMatrixOnHost(float *A, float *B, float *C, const int nx, const int ny);
void sumArraysOnHost(float * A, float *B, float *C, const int N);
void sumArraysOnHost(float *A, float *B, float *C, const int n, int offset);
void printData(char *msg, int *in,  const int size);
int recursiveReduce(int * data, int const size);

/*#define CHECK(call)                                                            \
{                                                                              \
    //const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        //fprintf(stderr, "code: %d, reason: %s\n", error,                       \
        //        cudaGetErrorString(error));                                    \
    }                                                                          \
}*/
#define CHECK(call)                                                            \
{                                                                              \
}
