#set -x
FILENAMES=(\
1-hipMalloc\ 
2-hipMemcpy\ 
4-hipKernel\ 
10-stream-2p\ 
)

TEST_MODE=0
QUICK_MODE=0
if [[ $QUICK_MODE -eq 1 ]] ; then
FILENAMES=(\
4-hipKernel\ 
)
fi

for FILENAME in ${FILENAMES[@]}; do
    LOG_DIR=log/$FILENAME
    BIN_DIR=bin
    echo Making directory $LOG_DIR
    mkdir $LOG_DIR -p 
    mkdir $BIN_DIR -p
    datasize_MB=256
    datasize_MB512=512
    declare -a SUB_DIR_SUFFIXES=(""  "-no-sdma" "-no-copy" "-timer" "-datasize_$datasize_MB" "-datasize_$datasize_MB-no-sdma" "-datasize_$datasize_MB512" "-datasize_$datasize_MB512-no-sdma")  
    index=0
    hipcc  $FILENAME.cpp  -o $BIN_DIR/$FILENAME.out

    for envvar in "" "HSA_ENABLE_SDMA=0" "nocopy=1" "timer=1" "datasize=$datasize_MB" "datasize=$datasize_MB HSA_ENABLE_SDMA=0" "datasize=$datasize_MB512" "datasize=$datasize_MB512 HSA_ENABLE_SDMA=0" ; do
        echo "==============================================="
        SUB_LOG_DIR=$LOG_DIR/$FILENAME${SUB_DIR_SUFFIXES[$index]}
        echo "SUB_LOG_DIR: $SUB_LOG_DIR"
        mkdir -p $SUB_LOG_DIR
        echo "envvar: $envvar"

        for envvar_prev_curr in ${envvar_prev_arr[@]}; do
            if [[ $envvar_prev_curr ]] ; then 
                echo "unsetting $envvar_prev_curr..."
                unset $envvar_prev_curr ; 
            fi
        done

        # reset the previus env var array.

        envvar_prev_arr=()

        export AMD_LOG_LEVEL=4 
        envvar_arr=($envvar)
        echo "envvar_arr: ${envvar_arr[@]}"
        for envvar_curr in ${envvar_arr[@]} ; do
            #echo "-------------------------"
            echo "exporting $envvar_curr..."
            export $envvar_curr
            envvar_prev=`echo $envvar_curr | cut -d '=' -f1`
            #echo "Adding $envvar_prev to envvar_prev_arr..."
            envvar_prev_arr+=($envvar_prev)
        done

        if [[ $TEST_MODE -eq 0 ]] ; then
            ./$BIN_DIR/$FILENAME.out 2>&1 | tee $SUB_LOG_DIR/$FILENAME.AMD_LOG_LEVEL.4.log
            rocprof --sys-trace -d ./$SUB_LOG_DIR/ ./$BIN_DIR/$FILENAME.out
            mv results* ./$SUB_LOG_DIR/
        fi

        echo "envvar_prev_arr: ${envvar_prev_arr[@]}"
        index=$((index+1))
    done
done
exit 0
