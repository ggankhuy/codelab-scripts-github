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
PATH_PFX_RESET_SW_SCRIPT="/home/mac-hq-02/tlee/G_s_iotool/iotools-1.5/reset_switch-2"
CONFIG_FAST_RUN=0
CONFIG_USE_MONITOR_SH=1

function warm_reboot () {
    echo "sleeping for $CONFIG_INTERVAL_SLEEP seconds..."

    sshpass -p $CONFIG_HOST_PASSWORD ssh -o StrictHostKeyChecking=no $CONFIG_HOST_USERNAME@$CONFIG_HOST_IP "reboot" 
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
}    

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

rm -rf $LOG_DIR/*
mkdir -p $LOG_DIR

i=0

echo "rebooting $CONFIG_HOST_IP"
if [[ $CONFIG_FAST_RUN -eq 0 ]] ; then 
    warm_reboot 
fi

sshpass -p $CONFIG_HOST_PASSWORD ssh -o StrictHostKeyChecking=no $CONFIG_HOST_USERNAME@$CONFIG_HOST_IP "mkdir /log ; ls -l /"
if [[ $? -ne 0 ]] ; then echo "ssh into terminal is failing, can not continue..." ; fi

while  [[ $i -le $CONFIG_ITERATION ]] 
do
    #1.  Boot mactruck with V340 and libgv
    #5.  Boot mactruck with V340 and libgv

    echo "Loading libgv..."
    sshpass -p $CONFIG_HOST_PASSWORD ssh -o StrictHostKeyChecking=no $CONFIG_HOST_USERNAME@$CONFIG_HOST_IP "dmesg --clear ; modprobe gim ; dmesg" | tee -a $LOG_DIR/gimload.iter.$i.log 

    #2.  Start Monitor deamon running in background in parallel, monitor all GPUs.
    #6.  Start Monitor Deamon running in background in parallel, monitor all GPUs.

    # Disabling for now. nohup bash -c still outputs to client ssh when issued on target ssh. This prevents the scripts from continuing.

    if [[ $CONFIG_USE_MONITOR_SH -eq 1 ]] ; then
        sshpass -p amd1234 scp -C -v -r -o StrictHostKeyChecking=no ./monitor.sh $CONFIG_HOST_USERNAME@$CONFIG_HOST_IP:/root/
        sshpass -p $CONFIG_HOST_PASSWORD ssh -o StrictHostKeyChecking=no $CONFIG_HOST_USERNAME@$CONFIG_HOST_IP "nohup /root/monitor.sh > /log/monitor.sh &" 
    else
        #sshpass -p $CONFIG_HOST_PASSWORD ssh -o StrictHostKeyChecking=no $CONFIG_HOST_USERNAME@$CONFIG_HOST_IP "rm /log/monitor.log ; nohup bash -c 'while true ; do $PATH_MONITOR  >> /log/monitor.log ; sleep 0.00002 ; done ' &" 
        #sshpass -p $CONFIG_HOST_PASSWORD ssh -o StrictHostKeyChecking=no $CONFIG_HOST_USERNAME@$CONFIG_HOST_IP "nohup bash -c 'while true ; ' &" 
        sshpass -p $CONFIG_HOST_PASSWORD ssh -o StrictHostKeyChecking=no $CONFIG_HOST_USERNAME@$CONFIG_HOST_IP "nohup bash -c 'i=0; while true ; do echo $i ; sleep 0.2 ; i=$((i+1)) ; done' &" 
    fi

    #7.  Use the SMI API amdgv_disable_gpu_access() on all GPUs.

    echo "check smi run"
    smi_run_flag=$((i%2))

    if [[ $smi_run_flag -eq 1 ]] ; then
        echo "iteration No. is odd: $i, running SMI flag." | tee  -a $LOG_DIR/smi.log
    fi

    if [[ $CONFIG_FAST_RUN -eq 0 ]] ; then
        echo "sleeping for 900 seconds."
        sleep 900
    else
        echo "sleeping for 30 seconds."
        sleep 30
    fi

    sshpass -p $CONFIG_HOST_PASSWORD ssh -o StrictHostKeyChecking=no $CONFIG_HOST_USERNAME@$CONFIG_HOST_IP "lspci -s 90:00.0 -x" | tee -a $LOG_DIR/90.0.0.before.sw.reset.$i.log 

    #3.  Apply the PFX switch reset script
    #8.  Apply the PFX switch reset script

    sshpass -p $CONFIG_HOST_PASSWORD ssh -o StrictHostKeyChecking=no $CONFIG_HOST_USERNAME@$CONFIG_HOST_IP "dmesg" | tee -a $LOG_DIR/post-900s-sleep.$i.log 

    sshpass -p $CONFIG_HOST_PASSWORD ssh -o StrictHostKeyChecking=no $CONFIG_HOST_USERNAME@$CONFIG_HOST_IP \
        "$PATH_PFX_RESET_SW_SCRIPT 0 140 0 0 >> /log/reset-2.log"
    sshpass -p $CONFIG_HOST_PASSWORD ssh -o StrictHostKeyChecking=no $CONFIG_HOST_USERNAME@$CONFIG_HOST_IP \
        "$PATH_PFX_RESET_SW_SCRIPT 0 148 0 0 >> /log/reset-3.log"

    sleep 5
    sshpass -p $CONFIG_HOST_PASSWORD ssh -o StrictHostKeyChecking=no $CONFIG_HOST_USERNAME@$CONFIG_HOST_IP "lspci -s 90:00.0 -x" | tee -a $LOG_DIR/90.0.0.fter.sw.reset.$i.log 
    sshpass -p $CONFIG_HOST_PASSWORD ssh -o StrictHostKeyChecking=no $CONFIG_HOST_USERNAME@$CONFIG_HOST_IP "dmesg" | tee -a $LOG_DIR/post-reset-switch.$i.log 

    #4.  Warm reboot. Please collect the host kernel logs
    #9.  Warm reboot. Please collect the host kernel logs
    warm_reboot
    sshpass -p $CONFIG_HOST_PASSWORD ssh -o StrictHostKeyChecking=no $CONFIG_HOST_USERNAME@$CONFIG_HOST_IP "dmesg" | tee -a $LOG_DIR/warm-reboot.iter.$i.log 

    i=$((i+1))
done
    

