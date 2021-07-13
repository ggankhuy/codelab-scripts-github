PATH_MONITOR=/usr/src/gim-2.1.3.G.20210413/smi-lib/examples/monitor/monitor
MONITOR_LOG=./monitor.log
i=0
echo "start" > $MONITOR_LOG
while true
    do
    $PATH_MONITOR >> $MONITOR_LOG
    sleep 0.00002
    i=$((i+1))
done

