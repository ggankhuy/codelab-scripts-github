builds=("-icd" "-icda gfx908:xnack-" "-icdn" "-icdna gfx908:xnack-")
#builds=("-icda gfx908:xnack-")
LOG_FOLDER=./log
mkdir -p $LOG_FOLDER
for (( i=0 ; i < ${#builds[@]} ; i++ ))  ;do
    rm -rf ./build
    rm -rf ./log
    echo $i: ${builds[$i]}
    start=$SECONDS
    ./install.sh ${builds[$i]} | tee $LOG_FOLDER/build.${builds[$i]}.log
    end=$SECONDS
    duration=$(( $end - $start ))
    echo "build: ${builds[$i]} - duration: $duration" >> $LOG_FOLDER/rocblas-build.log
done
