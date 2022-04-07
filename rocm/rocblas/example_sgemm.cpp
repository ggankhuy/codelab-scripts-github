#include <iostream>
#include <vector>
#include "hip/hip_runtime_api.h"
#include "rocblas.h"

using namespace std;

#define DIM1 1023
#define DIM2 1024
#define DIM3 1025

#define CHECK_HIP_ERROR(ERROR)                    \
    do                                            \
    {                                             \
        auto error = ERROR;                       \
        if(error != hipSuccess)                   \
        {                                         \
            fprintf(stderr,                       \
                    "error: '%s'(%d) at %s:%d\n", \
                    hipGetErrorString(error),     \
                    error,                        \
                    __FILE__,                     \
                    __LINE__);                    \
            exit(EXIT_FAILURE);                   \
        }                                         \
    } while(0)

// Extra macro so that macro arguments get expanded before calling Google Test

#define CHECK_ROCBLAS_ERROR(STATUS) CHECK_ROCBLAS_ERROR2(STATUS)
#define CHECK_ROCBLAS_ERROR2(STATUS) EXPECT_ROCBLAS_STATUS(STATUS, rocblas_status_success)
#define EXPECT_ROCBLAS_STATUS rocblas_expect_status

inline void rocblas_expect_status(rocblas_status status, rocblas_status expect)
{
    if(status != expect)
    {
        std::cerr << "rocBLAS status error: Expected " << rocblas_status_to_string(expect)
                     << ", received " << rocblas_status_to_string(status) << std::endl;
        if(expect == rocblas_status_success)
            exit(EXIT_FAILURE);
    }
}


int main()
{
    rocblas_operation transa = rocblas_operation_none, transb = rocblas_operation_transpose;
    float             alpha = 1.1, beta = 0.9;

    rocblas_int m = DIM1, n = DIM2, k = DIM3;
    rocblas_int lda, ldb, ldc, size_a, size_b, size_c;
    int         a_stride_1, a_stride_2, b_stride_1, b_stride_2;

    // Naming: da is in GPU (device) memory. ha is in CPU (host) memory
    vector<float> ha(size_a);
    vector<float> hb(size_b);
    vector<float> hc(size_c);
    vector<float> hc_gold(size_c);

    std::cout << "sgemm example" << std::endl;
    if(transa == rocblas_operation_none)
    {
        lda        = m;
        size_a     = k * lda;
        a_stride_1 = 1;
        a_stride_2 = lda;
        std::cout << "N";
    }
    else
    {
        lda        = k;
        size_a     = m * lda;
        a_stride_1 = lda;
        a_stride_2 = 1;
        std::cout << "T";
    }
    if(transb == rocblas_operation_none)
    {
        ldb        = k;
        size_b     = n * ldb;
        b_stride_1 = 1;
        b_stride_2 = ldb;
        std::cout << "N: ";
    }
    else
    {
        ldb        = n;
        size_b     = k * ldb;
        b_stride_1 = ldb;
        b_stride_2 = 1;
        std::cout << "T: ";
    }
    ldc    = m;
    size_c = n * ldc;

    // initial data on host
    srand(1);
    for(int i = 0; i < size_a; ++i)
    {
        ha[i] = rand() % 17;
    }
    for(int i = 0; i < size_b; ++i)
    {
        hb[i] = rand() % 17;
    }
    for(int i = 0; i < size_c; ++i)
    {
        hc[i] = rand() % 17;
    }
    hc_gold = hc;

    // allocate memory on device
    float *da, *db, *dc;
    CHECK_HIP_ERROR(hipMalloc(&da, size_a * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&db, size_b * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&dc, size_c * sizeof(float)));

    // copy matrices from host to device
    CHECK_HIP_ERROR(hipMemcpy(da, ha.data(), sizeof(float) * size_a, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(db, hb.data(), sizeof(float) * size_b, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dc, hc.data(), sizeof(float) * size_c, hipMemcpyHostToDevice));

    rocblas_handle handle;
    CHECK_ROCBLAS_ERROR(rocblas_create_handle(&handle));

    CHECK_ROCBLAS_ERROR(
        rocblas_sgemm(handle, transa, transb, m, n, k, &alpha, da, lda, db, ldb, &beta, dc, ldc));

    // copy output from device to CPU
    CHECK_HIP_ERROR(hipMemcpy(hc.data(), dc, sizeof(float) * size_c, hipMemcpyDeviceToHost));

    std::cout << "m, n, k, lda, ldb, ldc = " << m << ", " << n << ", " << k << ", " << lda
                 << ", " << ldb << ", " << ldc << std::endl;

    float max_relative_error = std::numeric_limits<float>::min();

    // calculate golden or correct result
    #ifdef CONFIG_GOLDEN_RESULT
    mat_mat_mult<float>(alpha,
                        beta,
                        m,
                        n,
                        k,
                        ha.data(),
                        a_stride_1,
                        a_stride_2,
                        hb.data(),
                        b_stride_1,
                        b_stride_2,
                        hc_gold.data(),
                        1,
                        ldc);

    for(int i = 0; i < size_c; i++)
    {
        float relative_error = (hc_gold[i] - hc[i]) / hc_gold[i];
        relative_error       = relative_error > 0 ? relative_error : -relative_error;
        max_relative_error
            = relative_error < max_relative_error ? max_relative_error : relative_error;
    }
    float eps       = std::numeric_limits<float>::epsilon();
    float tolerance = 10;
    if(max_relative_error != max_relative_error || max_relative_error > eps * tolerance)
    {
        std::cout << "FAIL: max_relative_error = " << max_relative_error << std::endl;
    }
    else
    {
        std::cout << "PASS: max_relative_error = " << max_relative_error << std::endl;
    }
    #endif

    CHECK_HIP_ERROR(hipFree(da));
    CHECK_HIP_ERROR(hipFree(db));
    CHECK_HIP_ERROR(hipFree(dc));
    CHECK_ROCBLAS_ERROR(rocblas_destroy_handle(handle));
    return EXIT_SUCCESS;
}
