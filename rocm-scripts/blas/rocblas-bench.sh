set -x
ROCBLAS_BENCH=./release/clients/staging/rocblas-bench
HIPBLASLT_BENCH=./release/hipblaslt-install/bin/hipblaslt-bench

BENCH=$ROCBLAS_BENCH
BENCH=$HIPBLASLT_BENCH
bench="rocblas"
bench="hipblas-lt"
GPU=MI100
GPU=MI250

LOG_DIR=./log/$GPU
mkdir -p $LOG_DIR

for type in f32_r, f16_r, bf16_r ; do
    for size in 4096 8192 ; do
        case "$BENCH" in
            $ROCBLAS_BENCH)
                size_params_val="-m $size -n $size -k $size"
                type_params_val="-r $type"
                calc_params_val="-f gemm"
                transposeA_param="--transposeA"
                transposeB_param="--transposeB"
                ;;
            $HIPBLASLT_BENCH)
                size_params="-m $size -n $size -k $size"
                type_params="-r $type"
                calc_params=""
                transposeA_param="--transA"
                transposeB_param="--transB"
                ;;
        esac

        $BENCH $calc_params_val $size_params | tee $LOG_DIR/$bench-bench.$type.$size.NN.log
        $BENCH $calc_params_val $size_params  $transposeA_param T | tee $LOG_DIR/$bench-bench.$type.$size.TN.log
        $BENCH $calc_params_val $size_params  $transposeA_param T $transposeB_param T | tee $LOG_DIR/$bench-bench.$type.$size.TT.log
        $BENCH $calc_params_val $size_params  $transposeB_param T | tee $LOG_DIR/$bench-bench.$type.$size.NT.log
    done
done
