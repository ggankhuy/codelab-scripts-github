set -x
FILENAME=1-hipMalloc
FILENAME=2-hipMemcpy
#FILENAME=2a-nohipMemcpy
#FILENAME=4-hipKernel
#FILENAME=4a-2xhipKernel 
LOG_DIR=log/$FILENAME
echo Making directory $LOG_DIR
mkdir $LOG_DIR -p 

declare -a SUB_DIR_SUFFIXES=(""  "no-sdma" "no-copy" "timer")  

index=0
for envvar in "" "HSA_ENABLE_SDMA=0" "nocopy=1" "timer=1" ; do
    echo "==============================================="

    SUB_LOG_DIR=$LOG_DIR/$FILENAME-${SUB_DIR_SUFFIXES[$index]}
    echo "SUB_LOG_DIR: $SUB_LOG_DIR"
    mkdir -p $SUB_LOG_DIR
    hipcc  $FILENAME.cpp  -o $FILENAME.out
    export AMD_LOG_LEVEL=4 
    export $envvar 
    ./$FILENAME.out 2>&1 | tee $SUB_LOG_DIR/$FILENAME.AMD_LOG_LEVEL.4.log
    rocprof --sys-trace -d ./$SUB_LOG_DIR/ ./$FILENAME.out
    mv results* ./$SUB_LOG_DIR/
    index=$((index+1))
done
exit 0
