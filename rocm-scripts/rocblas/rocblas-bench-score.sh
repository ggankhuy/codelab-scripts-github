DATE=`date +%Y%m%d-%H-%M-%S`
LOG_DIR=./log/$DATE
sudo mkdir -p $LOG_DIR

#Precision. Options: h,s,d,c,z,f16_r,f32_r,f64_r,bf16_r,f32_c,f64_c,i8_r,i32_r 

for iter_r in f16_r f32_r f64_r  ; do
    FILENAME_SCORE=./$LOG_DIR/$iter_r.score.csv
    echo "#$iter_r:" | sudo tee -a $FILENAME_SCORE
    for iter_n in 1024 2048 3048 4096 5120 8192 10240 16384 ; do

    iter_m=$iter_n
    iter_k=$iter_n
    iter_lda=$iter_n
    iter_ldb=$iter_n
    iter_ldc=$iter_n

    echo "-------"
    echo "type/m/n/k: $iter_f, $iter_n, $iter_m, $iter_k..."
    CMD="./rocblas-bench \
        -f gemm -r $iter_r --transposeA N --transposeB T \
        -m $iter_n -n $iter_m -k $iter_k --lda $iter_lda --ldb $iter_ldb --ldc $iter_ldc  --alpha -1 --beta 1 -i 3"
    echo $CMD
    sudo $CMD | grep "[NT],[NT],.*,.*" | sudo tee -a $FILENAME_SCORE

    done
done

