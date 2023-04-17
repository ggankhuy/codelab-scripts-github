LOG_DIR=./log
sudo mkdir -p $LOG_DIR
#Precision. Options: h,s,d,c,z,f16_r,f32_r,f64_r,bf16_r,f32_c,f64_c,i8_r,i32_r 
#for iter_r in f32_r f64_r i8_r i32_r  ; do
for iter_r in f16_r f32_r f64_r  ; do
    for iter_n in 1024 2048 3048 4096 5120 ; do

    iter_m=$iter_n
    iter_k=$iter_n
    echo "-------"
    echo "type/m/n/k: $iter_f, $iter_n, $iter_m, $iter_k..."
    FILENAME_SCORE=./$LOG_DIR/$iter_r.score.csv
    CMD="./rocblas-bench \
        -f gemm -r d --transposeA N --transposeB T \
        -m $iter_n -n $iter_m -k $iter_k --lda 5760 --ldb 5760 --ldc 5760    --alpha -1 --beta 1 -i 3"
    echo $CMD
    sudo $CMD | grep "[NT],[NT],.*,.*" | sudo tee -a $FILENAME_SCORE
    #sudo ./rocblas-bench \
    #-f gemm -r $iter_r  --transposeA N --transposeB T \
    #-m $iter_n -n $iter_m -k $iter_k --lda 5760 --ldb 5760 --ldc 5760  \
    #--alpha -1 --beta 1 -i 3 | grep "[NT],[NT],.*,.*"2>&1 | sudo tee -a $FILE_NAME_SCORE

    done
done

