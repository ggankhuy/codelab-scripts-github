#include <stdio.h>
#include <sys/time.h>
#include <stdbool.h>

double seconds();
void initialData(float * ip, int size);
void checkResult(float * hostRef, float * gpuRef, const int N);
void sumMatrixOnHost(float *A, float *B, float *C, const int nx, const int ny);
