set -x
ROCBLAS_BENCH=./release/clients/staging/rocblas-bench
#HIPBLASLT_BENCH=./release/hipblaslt-install/bin/hipblaslt-bench
CK_PROFILER=./bin/ckProfiler
#arg1: tensor operation (gemm=GEMM)
#arg2: data type (0=fp32, 1=fp16)
#arg3: matrix layout (0=NN, 1=NT, 2=TN, 3=TT)
#arg4: verification (0=no, 1=yes)
#arg5: initialization (0=no init, 1=integer value, 2=decimal value)
#arg6: print matrix value (0=no, 1=yes)
#arg7: run kernel # of times (>1)
#arg8 to 13: M, N, K, StrideA, StrideB, StrideC

BENCH=$ROCBLAS_BENCH
#BENCH=$HIPBLASLT_BENCH
BENCH=$CK_PROFILER
bench="rocblas"
#bench="hipblas-lt"

#GPU=MI100
GPU=MI250

LOG_DIR=./log/$GPU
mkdir -p $LOG_DIR

for type in f32_r f16_r bf16_r ; do
    for size in 512 1024 2048 4096 8192 ; do
        case "$BENCH" in
            $CK_PROFILER)
                calc_params_val="gemm"
                
                case "$type" in
                    "f32_r")
                        type_params_val=0
                        ;;
                    "f16_r")
                        type_params_val=1
                        ;;
                    "bf32_r")
                        type_params_val=2
                        ;;
                esac

                ck_specific_param="0 0 0 0 0"
                size_params_val="$size $size $size $size $size $size"
                ;;
            $ROCBLAS_BENCH)
                size_params_val="-m $size -n $size -k $size"
                type_params_val="-r $type"
                calc_params_val="-f gemm"
                transposeA_param="--transposeA"
                transposeB_param="--transposeB"
                transpose_val="T"
                ;;
            $HIPBLASLT_BENCH)
                size_params_val="-m $size -n $size -k $size"
                type_params_val="-r $type"
                calc_params_val=""
                transposeA_param="--transA"
                transposeB_param="--transB"
                transpose_val="T"
                ;;
        esac

        $BENCH $calc_params_val $type_params_val $ck_specific_param $size_params_val   | tee $LOG_DIR/$bench-bench.$type.$size.NN.log
        $BENCH $calc_params_val $type_params_val $ck_specific_param $size_params_val $transposeA_param $transpose_val   \
                                                                            | tee $LOG_DIR/$bench-bench.$type.$size.TN.log
        $BENCH $calc_params_val $type_params_val $ck_specific_param $size_params_val $transposeA_param $transpose_val $transposeB_param $transpose_val \
                                                                            | tee $LOG_DIR/$bench-bench.$type.$size.TT.log
        $BENCH $calc_params_val $type_params_val $ck_specific_param $size_params_val $transposeB_param $transpose_val  \
                                                                            | tee $LOG_DIR/$bench-bench.$type.$size.NT.log
    done
done
