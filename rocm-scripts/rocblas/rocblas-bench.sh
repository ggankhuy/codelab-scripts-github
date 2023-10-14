ROCBLAS_BENCH=./release/clients/staging/rocblas-bench
GPU=MI100
LOG_DIR=./log/$GPU
mkdir -p $LOG_DIR
#for type in f32_r, f16_r, bf16_r ; do
for type in bf16_r ; do
    for size in 4096 8192 ; do
        $ROCBLAS_BENCH -f gemm -m $size -n $size -k $size | tee $LOG_DIR/rocblas-bench.$type.$size.NN.log
        $ROCBLAS_BENCH -f gemm -m $size -n $size -k $size  --transposeA T | tee $LOG_DIR/rocblas-bench.$type.$size.TN.log
        $ROCBLAS_BENCH -f gemm -m $size -n $size -k $size  --transposeA T --transposeB T | tee $LOG_DIR/rocblas-bench.$type.$size.TT.log
        $ROCBLAS_BENCH -f gemm -m $size -n $size -k $size  --transposeB T | tee $LOG_DIR/rocblas-bench.$type.$size.NT.log
    done
done
