mkdir log -p
for FILENAME in sscal example_sgemm ; do
    echo "------- running $FILENAME .... ---------"
    ROCBLAS_LAYER=6 ./build/$FILENAME.out 2>&1 | tee log/ROCBLAS_LAYER6.$FILENAME.log
    TENSILE_DB=0x08000 ROCBLAS_LAYER=6 ./build/$FILENAME.out 2>&1 | tee log/TENSILE_DB.ROCBLAS_LAYER6.$FILENAME.log
done
