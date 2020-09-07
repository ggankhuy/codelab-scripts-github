DIRNAME=161384191-result
mkdir $DIRNAME

p1=$1

if [[ -z $1 ]] ; then
	echo "VM name not specified, defaulting to vats-test-01..."
	VM_NAME=vats-test-01
else
	echo "VM name: $p1..."
	sleep 5
	VM_NAME=$p1
fi

for i in {0..3}
do
	echo "start $VM_NAME"
	virsh start vats-test-01
	sleep 30
	VM_IP=`virsh domifaddr $VM_NAME | grep ipv4 | tr -s ' ' | cut -d ' ' -f5 | cut -d '/' -f1`
	echo "VM_IP obtained: $VM_IP" ; sleep 1
	sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$VM_IP 'dmesg > /tmp/dmesg'
	sshpass -p amd1234 scp -o StrictHostKeyChecking=no root@$VM_IP:/tmp/dmesg ./$DIRNAME/dmesg-iter-$i.log
	sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$VM_IP "dmesg --clear"

	pushd ..
	echo "running xgemm on guest $VM_IP..." ; sleep 1 
	 ./xgemm-run-on-guest.sh $VM_IP 0	
	popd
	echo "shutdown $VM_NAME" 
	virsh shutdown $VM_NAME & 	

	shutdown_time=0
	WAIT_INTERVAL=5
	for j in {0..10} ; do
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
