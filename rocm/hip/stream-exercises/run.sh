#for FILENAME in 1-2-gpu-serial 2-2-stream-parallel 3-2-stream-parallel-barrier-and 4-2+1-stream-parallel-barrier-and; do
for FILENAME in 5-stream-event; do
    
    LOG_FOLDER=log/$FILENAME/
    mkdir $LOG_FOLDER -p
    hipcc $FILENAME.cpp -o $FILENAME.out && AMD_LOG_LEVEL=4 ./$FILENAME.out 2>&1 | tee $LOG_FOLDER/$FILENAME.AMD_LOG_LEVEL.4.log
    if [[ $? -eq 0 ]] ; then rocprof --sys-trace $FILENAME.out else echo "compile with hipcc failed." ; exit 1 ; fi
    #rocprof --hip-trace $FILENAME.out
    if [[ $? -eq 0 ]] ; then mv results.* *.csv $LOG_FOLDER else echo "trace failed." ; exit 1 ; fi
done
