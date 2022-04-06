//https://github.com/ROCmSoftwarePlatform/rocBLAS
#include <iostream>
#include <vector>
#include "hip/hip_runtime_api.h"
#include "rocblas.h"

using namespace std;

int main()
{
    rocblas_int n = 10240;
    float alpha = 10.0;

    vector<float> hx(n);
    vector<float> hz(n);
    float* dx;

    rocblas_handle handle;
    rocblas_create_handle(&handle);

    // allocate memory on device
    hipMalloc(&dx, n * sizeof(float));

    // Initial Data on CPU,
    srand(1);
    for( int i = 0; i < n; ++i )
    {
        hx[i] = rand() % 10 + 1;  //generate a integer number between [1, 10]
    }

    // copy array from host memory to device memory
    hipMemcpy(dx, hx.data(), sizeof(float) * n, hipMemcpyHostToDevice);

    // call rocBLAS function
    rocblas_status status = rocblas_sscal(handle, n, &alpha, dx, 1);

    // check status for errors
    if(status == rocblas_status_success)
    {
        cout << "status == rocblas_status_success" << endl;
    }
    else
    {
        cout << "rocblas failure: status = " << status << endl;
    }

    // copy output from device memory to host memory
    hipMemcpy(hx.data(), dx, sizeof(float) * n, hipMemcpyDeviceToHost);

    hipFree(dx);
    rocblas_destroy_handle(handle);
    return 0;
}

