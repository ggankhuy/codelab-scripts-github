set -x
DATE=`date +%Y%m%d-%H-%M-%S`
cd build
rm -rf ../log/*
mkdir -p ../log/$DATE/
LOG_FOLDER=../log/$DATE/
ls -l ../log/
sleep 3

for i in sscal example_sgemm example_sgemm_strided_batched; do
    echo "======================" | tee $LOG_FOLDER/ROCBLAS_LAYER.4.TENSILE_DB.8000.$i.log
    ROCBLAS_LAYER=5 TENSILE_DB=8000 ./$i.out 2>&1 | tee -a $LOG_FOLDER/ROCBLAS_LAYER.5.TENSILE_DB.8000.$i.log
    echo "----------------------" | tee -a $LOG_FOLDER/ROCBLAS_LAYER.4.TENSILE_DB.8000.$i.log
    ROCBLAS_LAYER=4 TENSILE_DB=8000 ./$i.out 2>&1 | tee -a $LOG_FOLDER/ROCBLAS_LAYER.4.TENSILE_DB.8000.$i.log
done

cd ..
