FILENAME=file1.cpp
FILENAME=file2.cpp
FILENAME=file2a.cpp
mkdir log
hipcc $FILENAME && AMD_LOG_LEVEL=4  ./a.out 2>&1 | tee log/$FILENAME.AMD_LOG_LEVEL.4.log
