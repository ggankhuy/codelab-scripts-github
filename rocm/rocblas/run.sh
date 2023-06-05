<<<<<<< HEAD
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
=======
mkdir log -p
for FILENAME in sscal example_sgemm ; do
    echo "------- running $FILENAME .... ---------"
    ROCBLAS_LAYER=6 ./build/$FILENAME.out 2>&1 | tee log/ROCBLAS_LAYER6.$FILENAME.log
    TENSILE_DB=0x08000 ROCBLAS_LAYER=6 ./build/$FILENAME.out 2>&1 | tee log/TENSILE_DB.ROCBLAS_LAYER6.$FILENAME.log
done
>>>>>>> dev-gpu
