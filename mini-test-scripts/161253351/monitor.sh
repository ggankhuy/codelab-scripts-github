CONFIG_DURATION_HOURS=8
CONFIG_DURATION_SEC=$((CONFIG_DURATION_HOURS * 3600))
CONFIG_INTERVAL_SEC=5
CONFIG_LOOP_TIME=$(($CONFIG_DURATION_SEC / $CONFIG_INTERVAL_SEC))
LOG_FILE=monitor.log
CONFIG_ENABLE_LOG=0

if [[ $CONFIG_ENABLE_LOG -eq 1 ]] ; then
    echo "Running for Loop: $CONFIG_LOOP_TIME, Duration (sec): $CONFIG_DURATION_SEC" | tee $LOG_FILE
else
    echo "Running for Loop: $CONFIG_LOOP_TIME, Duration (sec): $CONFIG_DURATION_SEC"
fi

cd /usr/src/gim-2.0.1.G.20201023/smi-lib/examples/monitor 
make

echo "starting loop"

if [[ $CONFIG_ENABLE_LOG -eq 1 ]] ; then
    for i in $(seq 0 $CONFIG_DURATION_SEC) ; do
            echo "Loop No: $i" >> $LOG_FILE
            sleep $CONFIG_INTERVAL_SEC;
           ./monitor >> $LOG_FILE
    done
else
    for i in $(seq 0 $CONFIG_DURATION_SEC) ; do
            echo "Loop No: $i"
            sleep $CONFIG_INTERVAL_SEC;
           ./monitor
    done
fi
