TEST_MODE=0
builds=("-icd" "-icda gfx908:xnack-" "-icdn" "-icdna gfx908:xnack-")

if  [[ $TEST_MODE -eq 1 ]] ; then
    builds=("-icdn" "-icdna gfx908:xnack-")
fi
LOG_FOLDER=./log
rm -rf ./build
rm -rf ./$LOG_FOLDER
mkdir -p $LOG_FOLDER
for (( i=0 ; i < ${#builds[@]} ; i++ ))  ;do
    CURR_LOG=$LOG_FOLDER/build.${builds[$i]}.log
    echo $i: ${builds[$i]}
    start=$SECONDS
    ./install.sh ${builds[$i]} | tee $CURR_LOG
    tree -f build | tee -a $CURR_LOG
    end=$SECONDS
    duration=$(( $end - $start ))
    echo "build: ${builds[$i]} - duration: $duration" >> $LOG_FOLDER/rocblas-build.log
done
