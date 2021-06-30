#1.  Boot mactruck with V340 and libgv
#2.  Start Monitor deamon running in background in parallel, monitor all GPUs.
#3.  Apply the PFX switch reset script
#4.  Warm reboot. Please collect the host kernel logs
#5.  Boot mactruck with V340 and libgv
#6.  Start Monitor Deamon running in background in parallel, monitor all GPUs.
#7.  Use the SMI API amdgv_disable_gpu_access() on all GPUs.
#8.  Apply the PFX switch reset script
#9.  Warm reboot. Please collect the host kernel logs

# default values.

ip=""
CONFIG_INTERVAL_SLEEP=600
CONFIG_HOST_IP=""
CONFIG_HOST_USERNAME="root"
CONFIG_HOST_PASSWORD="amd1234"
CONFIG_ITERATION=3
LOG_DIR=./log/
PATH_MONITOR="/home/mac-hq-02/tlee/GpuTools-2019-01-19/libsmi/monitor"
PATH_PFX_RESET_SW_SCRIPT="/home/mac-hq-02/tlee/G_s_iotool/iotools-1.5/reset_switch"

for var in "$@"
do
    if [[ ! -z `echo "$var" | grep "ip="` ]]  ; then
        echo "ip address: $var"
        CONFIG_HOST_IP=`echo $var | cut -d '=' -f2`
    fi
    if [[ ! -z `echo "$var" | grep "user="` ]]  ; then
        echo "ip address: $var"
        CONFIG_HOST_USERNAME=`echo $var | cut -d '=' -f2`
    fi
    if [[ ! -z `echo "$var" | grep "pw="` ]]  ; then
        echo "ip address: $var"
        CONFIG_HOST_PASSWORD=`echo $var | cut -d '=' -f2`
    fi

    if [[ ! -z `echo "$var" | grep "sleep="` ]]  ; then
        CONFIG_INTERVAL_SLEEP=`echo $var | cut -d '=' -f2`
    fi

    if [[ ! -z `echo "$var" | grep "iter="` ]]  ; then
        CONFIG_ITERATION=`echo $var | cut -d '=' -f2`
    fi

    if [[ ! -z `echo "$var" | grep "help"` ]]  ; then
        echo "Usage: $0 <parameters>."
        echo "Parameters:"
        echo "ip=<host IP address>"
        echo "user=<host username>"
        echo "pw=<host password>"
        echo "sleep=<wait time in seconds between reboots>"
        exit 0 ;
    fi
done

if [[ -z $CONFIG_HOST_IP ]] ; then echo "IP address of the target machine must be specified." ; exit 1 ; fi

#1.  Boot mactruck with V340 and libgv
#2.  Start Monitor deamon running in background in parallel, monitor all GPUs.
#3.  Apply the PFX switch reset script
#4.  Warm reboot. Please collect the host kernel logs
#5.  Boot mactruck with V340 and libgv
#6.  Start Monitor Deamon running in background in parallel, monitor all GPUs.
#7.  Use the SMI API amdgv_disable_gpu_access() on all GPUs.
#8.  Apply the PFX switch reset script
#9.  Warm reboot. Please collect the host kernel logs

ping -c 4 $CONFIG_HOST_IP
if [[ $? -ne 0 ]] ; then echo "Host must be running. Not pingable..." ; exit 1;  fi

mkdir -p $LOG_DIR

i=0
while  [[ $i -le $CONFIG_ITERATION ]] 
do
    echo $i
    echo "rebooting $CONFIG_HOST_IP"
    sshpass -p $CONFIG_HOST_PASSWORD ssh -o StrictHostKeyChecking=no $CONFIG_HOST_USERNAME@$CONFIG_HOST_IP "echo rebooting... > /tmp/reboot.log; reboot"
    echo "sleeping for $CONFIG_INTERVAL_SLEEP seconds..."

    while [[ $j -le $CONFIG_INTERVAL_SLEEP ]] 
    do
        sleep 30
        ping -c 4 $CONFIG_HOST_IP
        if [[ $? -eq 0 ]] ; then
            echo "Host is pingable. Not waiting until $CONFIG_INTERVAL_SLEEP. Will sleep for another 30 sec for ssh init."
            sleep 30
            break
        fi
    done
    
    ping -c 4 $CONFIG_HOST_IP
    if [[ $? -ne 0 ]] ; then echo "Host is not up after sleeping for $CONFIG_INTERVAL_SLEEP seconds. Try increasing sleep time." ; exit 1;  fi
    echo "Loading libgv..."
    sshpass -p $CONFIG_HOST_PASSWORD ssh -o StrictHostKeyChecking=no $CONFIG_HOST_USERNAME@$CONFIG_HOST_IP "dmesg --clear ; modprobe gim ; dmesg" | tee -a $LOG_DIR/gimload.iter.$i.log 
    sshpass -p $CONFIG_HOST_PASSWORD ssh -o StrictHostKeyChecking=no $CONFIG_HOST_USERNAME@$CONFIG_HOST_IP "nohup bash -c 'for i in {0..100} ; do echo iteration >> /tmp/monitor.log; $PATH_MONITOR  > /tmp/monitor.$i.log ; sleep 0.2 ; done '" 
    sshpass -p $CONFIG_HOST_PASSWORD ssh -o StrictHostKeyChecking=no $CONFIG_HOST_USERNAME@$CONFIG_HOST_IP "$PATH_PFX_RESET_SW_SCRIPT > /tmp/reset.log"
    i=$((i+1))
done
    

