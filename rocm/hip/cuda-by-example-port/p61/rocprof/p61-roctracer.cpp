#include <stdio.h>
#include "hip/hip_runtime.h"


#ifdef __cplusplus
#include <cstdlib>
using namespace std;
#else
#include <stdlib.h>
#endif

// roctx header file
#include <roctx.h>
// roctracer extension API
#include <roctracer_ext.h>
#include <roctracer_hip.h>
#include <roctracer_hcc.h>
#include <roctracer_hsa.h>
#include <roctracer_kfd.h>
#include <roctracer_roctx.h>

#include <unistd.h>
#include <sys/syscall.h>   /* For SYS_xxx definitions */

static inline uint32_t GetTid() { return syscall(__NR_gettid); }
static inline uint32_t GetPid() { return syscall(__NR_getpid); }

// 1. if N is set to up to 1024, then sum is OK.
// 2. Set N past the 1024 which is past No. of threads per blocks, and then all iterations of sum results in 
// even the ones within the block.

// 3. To circumvent the problem described in 2. above, since if N goes past No. of threads per block, we need multiple block launch.
// The trick is describe in p65 to use formula (N+127) / 128 for blocknumbers so that when block number starts from 1, it is 
// (1+127) / 128.

#define N 4194304
#define N 2048
#define MAX_THREAD_PER_BLOCK 1024

#ifdef __cplusplus
static thread_local const size_t msg_size = 512;
static thread_local char* msg_buf = NULL;
static thread_local char* message = NULL;
#else
static const size_t msg_size = 512;
static char* msg_buf = NULL;
static char* message = NULL;
#endif

__global__ void add( int * a, int * b, int * c ) {
    int tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x ;
    if (tid < N) 
        c[tid] = a[tid] + b[tid];
}    

void SPRINT(const char* fmt, ...) {
    if (msg_buf == NULL) {
        msg_buf = (char*) calloc(msg_size, 1);
        message = msg_buf;
    }

    va_list args;
    va_start(args, fmt);
    message += vsnprintf(message, msg_size - (message - msg_buf), fmt, args);
    va_end(args);
}
void SFLUSH() {
    if (msg_buf == NULL) abort();
    message = msg_buf;
    msg_buf[msg_size - 1] = 0;
    fprintf(stdout, "%s", msg_buf);
    fflush(stdout);
}

// Runtime API callback function
void api_callback(
        uint32_t domain,
        uint32_t cid,
        const void* callback_data,
        void* arg)
{
    (void)arg;

    if (domain == ACTIVITY_DOMAIN_ROCTX) {
        const roctx_api_data_t* data = (const roctx_api_data_t*)(callback_data);
        fprintf(stdout, "rocTX <\"%s pid(%d) tid(%d)\">\n", data->args.message, GetPid(), GetTid());
        return;
    }
    if (domain == ACTIVITY_DOMAIN_KFD_API) {
        const kfd_api_data_t* data = (const kfd_api_data_t*)(callback_data);
        fprintf(stdout, "<%s id(%u)\tcorrelation_id(%lu) %s pid(%d) tid(%d)>\n",
                roctracer_op_string(ACTIVITY_DOMAIN_KFD_API, cid, 0),
                cid,
                data->correlation_id,
                (data->phase == ACTIVITY_API_PHASE_ENTER) ? "on-enter" : "on-exit", GetPid(), GetTid());
        return;
    }
    const hip_api_data_t* data = (const hip_api_data_t*)(callback_data);
    SPRINT("<%s id(%u)\tcorrelation_id(%lu) %s pid(%d) tid(%d)> ",
        roctracer_op_string(ACTIVITY_DOMAIN_HIP_API, cid, 0),
        cid,
        data->correlation_id,
        (data->phase == ACTIVITY_API_PHASE_ENTER) ? "on-enter" : "on-exit", GetPid(), GetTid());
    if (data->phase == ACTIVITY_API_PHASE_ENTER) {
        switch (cid) {
            case HIP_API_ID_hipMemcpy:
                SPRINT("dst(%p) src(%p) size(0x%x) kind(%u)",
                    data->args.hipMemcpy.dst,
                    data->args.hipMemcpy.src,
                    (uint32_t)(data->args.hipMemcpy.sizeBytes),
                    (uint32_t)(data->args.hipMemcpy.kind));
                break;
            case HIP_API_ID_hipMalloc:
                SPRINT("ptr(%p) size(0x%x)",
                    data->args.hipMalloc.ptr,
                    (uint32_t)(data->args.hipMalloc.size));
                break;
            case HIP_API_ID_hipFree:
                SPRINT("ptr(%p)", data->args.hipFree.ptr);
                break;
            case HIP_API_ID_hipModuleLaunchKernel:
                SPRINT("kernel(\"%s\") stream(%p)",
                    hipKernelNameRef(data->args.hipModuleLaunchKernel.f),
                    data->args.hipModuleLaunchKernel.stream);
                break;
            default:
                break;
        }
    } else {
        switch (cid) {
            case HIP_API_ID_hipMalloc:
                SPRINT("*ptr(0x%p)", *(data->args.hipMalloc.ptr));
                break;
            default:
                break;
        }
    }
    SPRINT("\n");
    SFLUSH();
}

void init_tracing();
void start_tracing();
void stop_tracing();

void start_tracing() {
    printf("# START tracing");
    // Start
    roctracer_start();
}

// Stop tracing routine
void stop_tracing() {
#if HIP_API_ACTIVITY_ON
    roctracer_disable_domain_callback(ACTIVITY_DOMAIN_HIP_API);
    roctracer_disable_domain_callback(ACTIVITY_DOMAIN_HIP_API);
    roctracer_disable_domain_activity(ACTIVITY_DOMAIN_HIP_API);
#endif
#if HCC_API_ACTIVITY_ON
    roctracer_disable_domain_activity(ACTIVITY_DOMAIN_HCC_OPS);
#endif
#if HSA_API_ACTIVITY_ON
    roctracer_disable_domain_activity(ACTIVITY_DOMAIN_HSA_OPS);
#endif
#if KFD_API_ACTIVITY_ON
    roctracer_disable_domain_callback(ACTIVITY_DOMAIN_KFD_API);
#endif
#if ROCTX_API_ACTIVITY_ON
    roctracer_disable_domain_callback(ACTIVITY_DOMAIN_ROCTX);
#endif
    roctracer_flush_activity();
    printf("# STOP    \n");
}

void activity_callback(const char* begin, const char* end, void* arg) {
    const roctracer_record_t* record = (const roctracer_record_t*)(begin);
    const roctracer_record_t* end_record = (const roctracer_record_t*)(end);

    SPRINT("\tActivity records:\n");
    while (record < end_record) {
        const char * name = roctracer_op_string(record->domain, record->op, record->kind);
        SPRINT("\t%25s\tcorrelation_id(%3lu) time_ns(%lu:%lu -> %10lu)",
            name,
            record->correlation_id,
            record->begin_ns,
            record->end_ns, record->end_ns - record->begin_ns);
        if ((record->domain == ACTIVITY_DOMAIN_HIP_API) || (record->domain == ACTIVITY_DOMAIN_KFD_API)) {
            SPRINT(" process_id(%12u) thread_id(%u)",
                record->process_id,
                record->thread_id);
        } else if (record->domain == ACTIVITY_DOMAIN_HCC_OPS) {
            SPRINT(" device_id(%12d) queue_id(%lu)",
                record->device_id,
                record->queue_id);
            if (record->op == HIP_OP_ID_COPY) SPRINT(" bytes(0x%zx)", record->bytes);
        } else if (record->domain == ACTIVITY_DOMAIN_HSA_OPS) {
            SPRINT(" se(%u) cycle(%12lu) pc(%lx)",
                record->pc_sample.se,
                record->pc_sample.cycle,
                record->pc_sample.pc);
        } else if (record->domain == ACTIVITY_DOMAIN_EXT_API) {
            SPRINT(" external_id(%12lu)", record->external_id);
        } else {
            fprintf(stderr, "Bad domain %d\n\n", record->domain);
            abort();
        }
        SPRINT("\n");
        SFLUSH();

        roctracer_next_record(record, &record);
    }
}

// Init tracing routine
void init_tracing() {
    printf("# INIT #\n");
    // roctracer properties
    roctracer_set_properties(ACTIVITY_DOMAIN_HIP_API, NULL);
    // Allocating tracing pool
    roctracer_properties_t properties;
    memset(&properties, 0, sizeof(roctracer_properties_t));
    properties.buffer_size = 0x1000;
    properties.buffer_callback_fun = activity_callback;
    roctracer_open_pool(&properties);
#if HIP_API_ACTIVITY_ON
    // Enable HIP API callbacks
    roctracer_enable_domain_callback(ACTIVITY_DOMAIN_HIP_API, api_callback, NULL);
    // Enable HIP activity tracing
    roctracer_enable_domain_activity(ACTIVITY_DOMAIN_HIP_API);
#endif
#if HCC_API_ACTIVITY_ON
    roctracer_enable_domain_activity(ACTIVITY_DOMAIN_HCC_OPS);
#endif
#if HSA_API_ACTIVITY_ON
    // Enable PC sampling
    roctracer_enable_op_activity(ACTIVITY_DOMAIN_HSA_OPS, HSA_OP_ID_RESERVED1);
#endif
#if KFD_API_ACTIVITY_ON
    // Enable KFD API tracing
    roctracer_enable_domain_callback(ACTIVITY_DOMAIN_KFD_API, api_callback, NULL);
#endif
#if ROCTX_API_ACTIVITY_ON
    // Enable rocTX
    roctracer_enable_domain_callback(ACTIVITY_DOMAIN_ROCTX, api_callback, NULL);
#endif
}

int main (void) {
    int *a, *b, *c;
    int *dev_a, *dev_b, *dev_c;
    int stepSize;

	// allocate dev memory for N size for pointers declared earlier.

    printf("\nAllocating memory...(size %u array size of INT).\n", N );
    roctxMark("Allocating memory");
    //init_tracing();
    //start_tracing();

    a = (int*)malloc(N * sizeof(int));
    b = (int*)malloc(N * sizeof(int));
    c = (int*)malloc(N * sizeof(int));
	hipMalloc( (void**)&dev_a, N * sizeof(int));
	hipMalloc( (void**)&dev_b, N * sizeof(int));
	hipMalloc( (void**)&dev_c, N * sizeof(int));

	for (int i = 0; i < N; i++) {
		a[i] = i;
		b[i] = i+2;
		c[i] = 999;
	}

    roctxMark("Copy");
	// copy the initialized local memory values to device memory. 

    printf("\nCopy host to device...");
    roctxRangePush("Memcopy host to dev");
	hipMemcpy(dev_a, a, N * sizeof(int), hipMemcpyHostToDevice);
	hipMemcpy(dev_b, b, N * sizeof(int), hipMemcpyHostToDevice);
	hipMemcpy(dev_c, c, N * sizeof(int), hipMemcpyHostToDevice);
    roctxRangePop();

    const unsigned blocks = 512;
    const unsigned threadsPerBlock = 256;

	// invoke the kernel: 
	// block count: (N+127)/128
	// thread count: 128
    
    roctxRangePush("Launch kernel");
    hipLaunchKernelGGL(add, blocks, threadsPerBlock, 0, 0, dev_a, dev_b, dev_c);
    roctxRangePop();
    roctxRangePush("Memcopy dev to host");
    hipMemcpy(a, dev_a, N * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(b, dev_b, N * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(c, dev_c, N * sizeof(int), hipMemcpyDeviceToHost);
    roctxRangePop();

    stepSize =  N /20 ;	
	for (int i = 0; i < N; i+=stepSize) {
		printf("%d + %d = %d\n", a[i], b[i], c[i]);
	}

	hipFree(dev_a);
	hipFree(dev_b);
	hipFree(dev_c);
    free(a);
    free(b);
    free(c);
    //stop_tracing();
    return 0;
}
