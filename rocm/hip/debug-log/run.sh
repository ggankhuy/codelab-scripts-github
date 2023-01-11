FILENAME=1-hipMalloc
FILENAME=2-hipMemcpy
LOG_DIR=log/$FILENAME
echo Making directory $LOG_DIR
mkdir $LOG_DIR -p 
hipcc -D timer=0 $FILENAME.cpp  -o $FILENAME.out && AMD_LOG_LEVEL=4  ./$FILENAME.out 2>&1 | tee $LOG_DIR/$FILENAME.AMD_LOG_LEVEL.4.log
rocprof --sys-trace -d ./$LOG_DIR/ ./$FILENAME.out

hipcc -D timer=1 $FILENAME.cpp  -o $FILENAME-t.out && ./$FILENAME-t.out
mv results* ./$LOG_DIR/
