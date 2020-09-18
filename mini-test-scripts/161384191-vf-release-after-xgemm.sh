DIRNAME=161384191-result
mkdir $DIRNAME
CONFIG_VATS2_SUPPORT=1
CONFIG_ITERATIONS=10
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
		sleep 30
		VM_IP=`virsh domifaddr $VM_NAME | grep ipv4 | tr -s ' ' | cut -d ' ' -f5 | cut -d '/' -f1`
		if [[ $? -ne 0 ]] ; then
			echo "Failed to obtain IP,, skipping ..."
			continue
		fi
		echo "VM_IP obtained: $VM_IP" ; sleep 1
		sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$VM_IP 'dmesg > /tmp/dmesg'
		sshpass -p amd1234 scp -o StrictHostKeyChecking=no root@$VM_IP:/tmp/dmesg ./$DIRNAME/dmesg-iter-$k-vm-$i.log
		sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$VM_IP "dmesg --clear"

		pushd ..
		echo "running xgemm on guest $VM_IP..." ; sleep 1 
		 ./xgemm-run-on-guest.sh $VM_IP 0	
		popd
		echo "shutdown $VM_NAME" 
		virsh shutdown $VM_NAME & 	

		shutdown_time=0
		WAIT_INTERVAL=10
		for j in {0..30} ; do
			sleep $WAIT_INTERVAL
			echo "checking if $VM_NAME is shutdown ..."
			stat=`virsh list | grep $VM_NAME`
			if [[ -z $stat ]] ; then
				echo "shutdown of $VM_NAME is completed..."
				break
			else
				echo "shutdown of $VM_NAME is not completed, waiting more..."
			fi			
			shutdown_time=$(($shutdown_time+$WAIT_INTERVAL))
			echo "shutdown wait time so far: $shutdown_time..."
		done
		echo "shutdown_time for reboot No. $i: $shutdown_time" >> ./$DIRNAME/shutdown_times.log
	done
done
