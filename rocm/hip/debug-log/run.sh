FILENAME=hipMalloc
LOG_DIR=log/$FILENAME
echo Making directory $LOG_DIR
mkdir $LOG_DIR -p 
hipcc $FILENAME.cpp && AMD_LOG_LEVEL=4  ./a.out 2>&1 | tee $LOG_DIR/$FILENAME.AMD_LOG_LEVEL.4.log
rocprof --sys-trace -d ./$LOG_DIR/ ./a.out
mv results* ./$LOG_DIR/
