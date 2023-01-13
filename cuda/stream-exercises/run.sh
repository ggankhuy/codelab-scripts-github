#for FILENAME in 1-2-gpu-serial 2-2-stream-parallel 3-2-stream-parallel-barrier-and 4-2+1-stream-parallel-barrier-and 5-stream-event 6-2-gpu-2-stream-synchronize; do
#for FILENAME in 6-2-gpu-2-stream-synchronize ; do
#for FILENAME in 1-2-gpu-serial ; do
for FILENAME in 2-2-stream-parallel ; do
    
    LOG_FOLDER=log/$FILENAME/
    rm -rf $LOG_FOLDER/*
    mkdir $LOG_FOLDER -p
    nvcc $FILENAME.cu -o $FILENAME.out 2>&1 | tee $LOG_FOLDER/$FILENAME.compile.log
    tree -s $LOG_FOLDER/
    ./$FILENAME.out 2>&1 | tee -a $LOG_FOLDER/$FILENAME.run.log
done
