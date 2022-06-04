DOUBLE_BAR="========================================================"
SINGLE_BAR="--------------------------------------------------------"
LOG_DIR=log
mkdir -p $LOG_DIR

for var in "$@"
do
    if [[ ! -z `echo "$var" | grep "vmname="` ]]  ; then
        echo "VMNAME: $var"
        VM_NAME=`echo $var | cut -d '=' -f2`
    fi

    if [[ ! -z `echo "$var" | grep "iter="` ]]  ; then
        echo "No. of iteration: $var"
        ITER=`echo $var | cut -d '=' -f2`
    fi
done

if [[ $1 == '--help' ]] || [[ $1 == "" ]]   ; then
    echo "Usage: $0 <parameters>."
    echo "Parameters:"
    echo "vmname=<VM_NAME>"
    echo "iter=<No. of iteration of start/shutdown of VM_NAME>"
    exit 0 ;
fi

if [[ -z $VM_NAME ]] ; then echo "VM_NAME is empty!" ; exit 1; fi
if [[ -z $ITER ]] ; then echo "ITER is empty!" ; exit 1; fi

dmesg --clear

for (( n=0; n < $ITER; n++ ))  ; do
    echo $DOUBLE_BAR
    echo n: $n
    virsh shutdown $VM_NAME    
    sleep 5
    
    echo "Waiting for vm shutdown to complete..."

    SHUTDOWN_SUCCESS=1
    for ((i=0 ; i < 120 ; i ++ )) ; do
        echo -ne "."
        ret=`virsh list | grep $VM_NAME`
        #echo "virsh list | grep $VM_NAME output: $ret"
        
        if [[ -z $ret ]] ; then
            echo "1. done..."
            SHUTDOWN_SUCCESS=0
            break
        fi
        sleep 2
    done

    echo "Gathering dmeg..."
    dmesg > $LOG_DIR/dmesg.host.shutdown.$VM_NAME.loop-$n.log
    dmesg --clear

    if [[ $SHUTDOWN_SUCCESS -eq 1 ]] ; then echo "Shutdown of $VM_NAME was not successful on $n th loop..." ; exit 1 ; fi
    
    virsh start $VM_NAME    
    sleep 10

    VM_START_SUCCESS=1
    for ((i=0 ; i < 120 ; i ++ )) ; do
        VM_IP=`virsh domifaddr $VM_NAME | grep ipv4 | tr -s ' ' | cut -d ' ' -f5 | cut -d '/' -f1`
        ping -c 1 $VM_IP
        ret=$? 
        #echo "ret: $ret"
        if [[ $ret -eq 0 ]] ; then
            echo "2. done..."
            VM_START_SUCCESS=0
            sleep 3
            break
        fi
        echo -ne "."
        sleep 2
    done
    dmesg > $LOG_DIR/dmesg.host.shutdown.$VM_NAME.loop-$n.log
    dmesg --clear
    sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$VM_IP "dmesg" > $LOG_DIR/dmesg.guest.shutdown.$VM_NAME.loop-$n.log
    sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$VM_IP "dmesg --clear"

    if [[ $VM_START_SUCCESS -eq 1 ]] ; then echo "Start of $VM_NAME was not successful on $n th loop..." ; exit 1 ; fi
done

