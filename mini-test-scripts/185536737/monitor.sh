PATH_MONITOR="/home/mac-hq-02/tlee/GpuTools-2019-01-19/libsmi/monitor"

#nohup bash -c 'i=0; while true ; do echo $i ; sleep 0.2 ; i=$((i+1)) ; done' &
i=0
echo "start" > /log/monitor.log
while true 
    do 
#   echo $i >>  /log/monitor.log
    $PATH_MONITOR >> /log/monitor.log
    sleep 0.00002 
    i=$((i+1)) 
done
