for FILENAME in 2-gpu-serial 2-stream-parallel; do
#for FILENAME in 2-stream-parallel; do
    
    LOG_FOLDER=log/$FILENAME/
    mkdir $LOG_FOLDER -p
    hipcc $FILENAME.cpp -o $FILENAME.out && AMD_LOG_LEVEL=4 ./$FILENAME.out 2>&1 | tee $LOG_FOLDER/$FILENAME.AMD_LOG_LEVEL.4.log
    rocprof --sys-trace $FILENAME.out
    #rocprof --hip-trace $FILENAME.out
    mv results.* *.csv $LOG_FOLDER
done
