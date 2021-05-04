DOUBLE_BAR="========================================================"
SINGLE_BAR="--------------------------------------------------------"

fi
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

if [[ -z $VM_NAME ]] ; then echo "VM_NAME is empty!" ; exit 1; fi
if [[ -z $ITER ]] ; then echo "ITER is empty!" ; exit 1; fi

dmesg --clear

for (( n=0; n < $ITER; n++ ))  ; do
    echo $DOUBLE_BAR
    echo n: $n
    virsh shutdown $VM_NAME    
    sleep 15
    
    echo "Waiting for vm shutdown to complete..."

    SHUTDOWN_SUCCESS=1
    for ((i=0 ; i < 60 ; i ++ )) ; do
        ret=`virsh list | grep $VM_NAME`
        
        if [[ -z $ret ]] ; then
            echo "done..."
            break
            SHUTDOWN_SUCCESS=0
        fi
        echo -ne "."
        sleep 10
    done

    dmesg > dmesg.shutdown.$VM_NAME.loop-$n.log
    dmesg --clear

    if [[ $SHUTDOWN_SUCCESS -eq 1 ]] ; then echo "Shutdown of $VM_NAME was not successful on $n th loop..." ; exit 1 ; fi
    
    virsh start $VM_NAME    
    sleep 15

    VM_START_SUCCESS=1
    for ((i=0 ; i < 60 ; i ++ )) ; do
        VM_IP=`virsh domifaddr $VM_NAME | grep ipv4 | tr -s ' ' | cut -d ' ' -f5 | cut -d '/' -f1`
        ret=`ping -c 1 $VM_IP`
        if [[ -z $ret ]] ; then
            echo "done..."
            VM_START_SUCCESS=0
            sleep 3
            break
        fi
        echo -ne "."
        sleep 10
    done
    dmesg > dmesg.shutdown.host.$VM_NAME.loop-$n.log
    dmesg --clear
    sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$VM_IP "dmesg" > dmesg.shutdown.guest.$VM_NAME.loop-$n.log
    sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$VM_IP "dmesg --clear"`

    if [[ $VM_START_SUCCESS -eq 1 ]] ; then echo "Shutdown of $VM_NAME was not successful on $n th loop..." ; exit 1 ; fi
done

