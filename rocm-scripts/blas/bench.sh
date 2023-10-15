set -x
ROCBLAS_BENCH=./release/clients/staging/rocblas-bench
HIPBLASLT_BENCH=./release/hipblaslt-install/bin/hipblaslt-bench

BENCH=$ROCBLAS_BENCH
BENCH=$HIPBLASLT_BENCH
bench="rocblas"
bench="hipblas-lt"

#GPU=MI100
GPU=MI250

LOG_DIR=./log/$GPU
mkdir -p $LOG_DIR

for type in f32_r f16_r bf16_r ; do
    for size in 512 1024 2048 4096 8192 ; do
        case "$BENCH" in
            $ROCBLAS_BENCH)
                size_params_val="-m $size -n $size -k $size"
                type_params_val="-r $type"
                calc_params_val="-f gemm"
                transposeA_param="--transposeA"
                transposeB_param="--transposeB"
                ;;
            $HIPBLASLT_BENCH)
                size_params_val="-m $size -n $size -k $size"
                type_params_val="-r $type"
                calc_params_val=""
                transposeA_param="--transA"
                transposeB_param="--transB"
                ;;
        esac

#--precision |-r <value>    
#Precision. Options: h,s,d,c,z,f8_r, bf8_r, f16_r,f32_r,f64_r,bf16_r,f32_c,f64_c,i8_r,i32_r  (Default value is: f32_r)

        $BENCH $calc_params_val -r $type $size_params_val                       | tee $LOG_DIR/$bench-bench.$type.$size.NN.log
        $BENCH $calc_params_val -r $type $size_params_val $transposeA_param T   | tee $LOG_DIR/$bench-bench.$type.$size.TN.log
        $BENCH $calc_params_val -r $type $size_params_val $transposeA_param T $transposeB_param T \
                                                                                | tee $LOG_DIR/$bench-bench.$type.$size.TT.log
        $BENCH $calc_params_val -r $type $size_params_val $transposeB_param T   | tee $LOG_DIR/$bench-bench.$type.$size.NT.log
    done
done
