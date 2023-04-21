# make sure to install rocblas-devel (yum) or rocblas-dev (apt) package.
set -x
for i in sscal example_sgemm example_sgemm_strided_batched; do
    sudo hipcc -lrocblas  $i.cpp -o $i.out
    ROCBLAS_LAYER=5 TENSILE_DB=8000 ./$i.out 2>&1 | tee ../log/ROCBLAS_LAYER.5.TENSILE_DB.8000.$i.log
    ROCBLAS_LAYER=4 TENSILE_DB=8000 ./$i.out 2>&1 | tee ../log/ROCBLAS_LAYER.4.TENSILE_DB.8000.$i.log
done

chmod 755 *.out
cd ..
