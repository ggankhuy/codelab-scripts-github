DIRNAME=161384191-result
WAIT_INTERVAL=3
mkdir $DIRNAME
CONFIG_VATS2_SUPPORT=1
CONFIG_ITERATIONS=10
CONFIG_BY_PASS_XGEMM=1
DATE=`date +%Y%m%d-%H-%M-%S`
p0=$0
p1=$1
p2=$2
SINGLE_BAR="---------------------------------------------"
DOUBLE_BAR="============================================"
if [[ $p1 == "--help" ]] ; then
	clear
	echo "Usage: "
	echo "$p0 --help: display this help."
	echo "$p0 --all: run on all VM"
	echo "$p0 <VM_NAME>: run on specified VM <VM_NAME>"
	exit 0
fi

if [[ -z $p1 ]] ; then
	echo "VM name not specified, defaulting to vats-test-01..."
	VM_NAME=vats-test-01
else
	echo "VM name: $p1..."
	sleep 5
	VM_NAME=$p1
fi

if [[ $p1 -eq "all" ]] ; then
	if [[ $CONFIG_VATS2_SUPPORT -eq 1 ]] ; then
	    VM_GREP_PATTERN=vats
	else
	    VM_GREP_PATTERN=gpu
	fi
fi

echo "Backup host dmesg as dmesg-host-$DATE.log..."
dmesg >/$DIRNAME/ dmesg-host-$DATE.log
dmesg --clear
TOTAL_VMS=`virsh list --all | grep -i $VM_GREP_PATTERN | grep running | wc -l`

echo "TOTAL_VMS: $TOTAL_VMS"

for (( n=0; n < $TOTAL_VMS; n++ ))  ; do
    echo $DOUBLE_BAR
    echo n: $n
    GPU_INDEX=$n
    VM_INDEX=$(($n+1))
    echo "VM_INDEX: $VM_INDEX"

    VM_NAME=`virsh list --all |         grep $VM_GREP_PATTERN | head -$(($GPU_INDEX+1)) | tail -1  | tr -s ' ' | cut -d ' ' -f3`
    VM_NAMES[$n]=$VM_NAME
    VM_NO=`virsh list --all |         grep $VM_GREP_PATTERN | head -$(($GPU_INDEX)) | tail -1  | tr -s ' ' | cut -d ' ' -f2`
    VM_IP=`virsh domifaddr $VM_NAME | grep ipv4 | tr -s ' ' | cut -d ' ' -f5 | cut -d '/' -f1`
    VM_IPS[$n]=$VM_IP

    echo VM_NAME: $VM_NAME, VM_INDEX: $VM_INDEX, VM_NO: $VM_NO, GPU_INDEX: $GPU_INDEX, VM_IP: $VM_IP
    sleep 1

    if [[ $CONFIG_BY_PASS_XGEMM -ne 0 ]] ; then
        sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$VM_IP 'mkdir /xgemm'
        sshpass -p amd1234 scp -C -v -r -o StrictHostKeyChecking=no  /xgemm/* root@$VM_IP:/xgemm/
    fi
    if [[ $? -ne 0 ]] ; then echo "Unable to find xgemm on host."; exit 1; fi
done

echo $DOUBLE_BAR
echo "Started all VMs, running test now..."
echo $DOUBLE_BAR
for (( k=0; k < $CONFIG_ITERATIONS; k++ ))
do
	echo $DOUBLE_BAR
	echo "ITERATION $k of $CONFIG_ITERATIONS"
	echo $DOUBLE_BAR

	for i in ${VM_NAMES[@]}
	do
		echo $SINGLE_BAR
		echo "Launching xgemm on VM_NAME: $i, ITERATION $k"
		echo $SINGLE_BAR
		echo "start $i"
		virsh start $i

		timeout=0
                for j in {0..10} ; do
                        sleep $WAIT_INTERVAL
			VM_IP=`virsh domifaddr $i | grep ipv4 | tr -s ' ' | cut -d ' ' -f5 | cut -d '/' -f1`
                        if [[ ! -z $VM_IP ]] ; then
				ping -c 10 $VM_IP
				if [[ $? -eq 0 ]]  ; then
	                                break
				else 
					echo "Can not ping, extending timeout..."
				fi
                        else
                                echo "Can not get IP, waiting more..."
                        fi
                        timeout=$(($timeout+$WAIT_INTERVAL))
                        echo "wait time so far for obtaining IP for $i...: $timeout seconds."
                done

		if [[ -z $VM_IP ]] ; then
			echo "Failed to obtain IP, skipping VM $i..."
			continue
		fi
		echo "VM_IP obtained: $VM_IP" ; sleep 1
		sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$VM_IP 'dmesg > /tmp/dmesg'
		sshpass -p amd1234 scp -o StrictHostKeyChecking=no root@$VM_IP:/tmp/dmesg ./$DIRNAME/dmesg-iter-$k-vm-$i.log
		sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$VM_IP "dmesg --clear"

		pushd ..

		if [[ $CONFIG_BY_PASS_XGEMM -eq 0 ]] ; then
			echo "running xgemm on guest $VM_IP..." ; sleep 1 
			 ./xgemm-run-on-guest.sh $VM_IP 0	
		else
			echo "Bypassing xgemm..."
		fi
		popd
		echo "shutdown $i" 
		virsh shutdown $i & 	

		shutdown_time=0
		for j in {0..30} ; do
			sleep $WAIT_INTERVAL
			echo "checking if $i is shutdown ..."
			stat=`virsh list | grep $i`
			if [[ -z $stat ]] ; then
				echo "shutdown of $i is completed..."
				shutdown_time=$(($shutdown_time+$WAIT_INTERVAL))
				break
			else
				echo "shutdown of $i is not completed, waiting more..."
			fi			
			shutdown_time=$(($shutdown_time+$WAIT_INTERVAL))
			echo "shutdown wait time so far: $shutdown_time..."
		done
		echo "shutdown_time for vm: $i, iteration No. $k: $shutdown_time" >> ./$DIRNAME/shutdown_times.log
	done

	echo "Saving host dmesg..."
	touch ./$DIRNAME/dmesg-iter-$k-host.log
	dmesg > ./$DIRNAME/dmesg-iter-$k-host.log
	dmesg --clear
done

echo "End of t est: Turning back on all vm-s..."
for i in ${VM_NAMES[@]}
	virsh start $i 
done

virsh list
