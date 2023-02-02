set -x
FILENAME=1-hipMalloc
FILENAME=2-hipMemcpy
#FILENAME=2a-nohipMemcpy
#FILENAME=4-hipKernel
#FILENAME=4a-2xhipKernel 
LOG_DIR=log/$FILENAME
BIN_DIR=bin
echo Making directory $LOG_DIR
mkdir $LOG_DIR -p 
mkdir $BIN_DIR -p
datasize_MB=1024
declare -a SUB_DIR_SUFFIXES=(""  "-no-sdma" "-no-copy" "-timer" "-datasize_$datasize_MB" "-datasize_$datasize_MB-no-sdma")  

index=0
for envvar in "" "HSA_ENABLE_SDMA=0" "nocopy=1" "timer=1" "datasize=$datasize_MB" "datasize=$datasize_MB HSA_ENABLE_SDMA=0" ; do
    echo "==============================================="
    SUB_LOG_DIR=$LOG_DIR/$FILENAME${SUB_DIR_SUFFIXES[$index]}
    echo "SUB_LOG_DIR: $SUB_LOG_DIR"
    mkdir -p $SUB_LOG_DIR
    hipcc  $FILENAME.cpp  -o $BIN_DIR/$FILENAME.out
    if [[ $envvar_prev ]] ; then unset $envvar_prev ; fi
    export AMD_LOG_LEVEL=4 
    export $envvar 
    ./$BIN_DIR/$FILENAME.out 2>&1 | tee $SUB_LOG_DIR/$FILENAME.AMD_LOG_LEVEL.4.log
    rocprof --sys-trace -d ./$SUB_LOG_DIR/ ./$BIN_DIR/$FILENAME.out
    mv results* ./$SUB_LOG_DIR/
    envvar_prev=`echo $envvar | cut -d '=' -f1`
    echo "envvar_prev: $envvar_prev"
    index=$((index+1))
done
exit 0
