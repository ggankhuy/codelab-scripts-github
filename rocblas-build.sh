#builds=("-icd" "-icda gfx908:xnack-" "-icdn" "-icdna gfx908:xnack-")
builds=("-icdn" "-icdna gfx908:xnack-")
LOG_FOLDER=./log
mkdir -p $LOG_FOLDER
rm -rf ./build
rm -rf ./$LOG_FOLDER
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
