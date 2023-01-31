FILENAME=1-hipMalloc
FILENAME=2-hipMemcpy
#FILENAME=2a-nohipMemcpy
#FILENAME=4-hipKernel
#FILENAME=4a-2xhipKernel 
FILENAME_ORIG=$FILENAME
LOG_DIR=log/$FILENAME
echo Making directory $LOG_DIR
mkdir $LOG_DIR -p 

# vanilla run.

FILENAME=$FILENAME_ORIG
SUB_LOG_DIR=$LOG_DIR/$FILENAME
mkdir -p $SUB_LOG_DIR
hipcc -D timer=0 $FILENAME_ORIG.cpp  -o $FILENAME.out && \
AMD_LOG_LEVEL=4  ./$FILENAME.out 2>&1 | tee $SUB_LOG_DIR/$FILENAME.AMD_LOG_LEVEL.4.log
rocprof --sys-trace -d ./$SUB_LOG_DIR/ ./$FILENAME.out
mv results* ./$SUB_LOG_DIR/

# nosdma

FILENAME=$FILENAME-no-sdma
SUB_LOG_DIR=$LOG_DIR/$FILENAME
mkdir -p $SUB_LOG_DIR
export HSA_ENABLE_SDMA=0 
export AMD_LOG_LEVEL=4  
 ./$FILENAME.out 2>&1 | tee $SUB_LOG_DIR/$FILENAME-no-sdma.AMD_LOG_LEVEL.4.log
rocprof --sys-trace -d ./$SUB_LOG_DIR/ ./$FILENAME.out
mv results* ./$SUB_LOG_DIR/

# no memCpy 

FILENAME=$FILENAME_ORIG-no-copy
SUB_LOG_DIR=$LOG_DIR/$FILENAME
hipcc -D nocopy=1 $FILENAME_ORIG.cpp  -o $FILENAME.out && \
export AMD_LOG_LEVEL=4  
 ./$FILENAME.out 2>&1 | tee $SUB_LOG_DIR/$FILENAME.AMD_LOG_LEVEL.4.log
rocprof --sys-trace -d ./$SUB_LOG_DIR/ ./$FILENAME.out
mv results* ./$SUB_LOG_DIR/

# timer.

hipcc -D timer=1 $FILENAME_ORIG.cpp  -o $FILENAME-t.out && ./$FILENAME-t.out


