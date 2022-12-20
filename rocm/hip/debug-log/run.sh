FILENAME=file1.cpp
FILENAME=file2.cpp
FILENAME=file2a.cpp
FILENAME=file2b.cpp
FILENAME=file2c.cpp
FILENAME=file2d.cpp
FILENAME=file2e.cpp
FILENAME=file2f.cpp
FILENAME=file3.cpp
LOG_DIR=log-$FILENAME
mkdir $LOG_DIR
hipcc $FILENAME && AMD_LOG_LEVEL=4  ./a.out 2>&1 | tee $LOG_DIR/$FILENAME.AMD_LOG_LEVEL.4.log
rocprof --sys-trace -d $LOG_DIR/  ./a.out
