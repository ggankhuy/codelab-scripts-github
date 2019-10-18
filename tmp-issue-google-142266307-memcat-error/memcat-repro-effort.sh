6#  This script assumes all the VMs on either ixt39 or ixt70 is created afresh using autotest scripts i.e.
#  runtest 6 from autotest.
#  This script assumes the ens3 interface is used for streaming client and server. If the interface name is different
#  or absent, the result is unpredictable.
#  This script assumes the VM name is structured as debian-drop-<month>-<date>-debian-gpu<No>-vf<No> format 
#  using auto-test script. If the VM name is structured differently in any way, the result is unpredictable.
#  If there are VMs that are created for multiple drops, either running or shutdown, the result is extremely unpredictable. 
#  
#  Steps this tool takes:
#  Count all vms.
#  Load gim.
#  Start default network.
#  Turn on all vms.
#  Log on to each vm through ssh (determine ip using virsh domifaddr <vmno>
#  update /etc/network/interfaces with static ip from pool.
#  IP address range: 10.216.66.67-78.
#  Assignment:
#  
#  ixt39  4vm-s / 4 gpu-s, 10.216.66.67-70.
#  ixt70  8vm-s / 8 gpu-s, 10.216.66.71-78.

# Turn off all vm-s
# Set vm vcpu-s to 8 as standard.
# Turn on all VM-s 

source ./common.sh

DOUBLE_BAR="========================================================"
SINGLE_BAR="--------------------------------------------------------"
CONFIG_SUPPORT_MEMCAT=1
CONFIG_MEMCAT_SRC_DIR=/root/memcat/
CONFIG_MEMCAT_DST_DIR=/memcat/
CONIG_LOOP_TEST_NO=3
CONFIG_SET_VCPUCOUNT=0
SETUP_GAME_VM_CLIENT=setup-game-vm-client.sh
DATE=`date +%Y%m%d-%H-%M-%S`

#	Following setting requires great diligence from user of this script. When running flag is set 
#	The  TOTAL_VMS will only count the running VM-s. This could be useful to not count non-running VM
#	which is irrelevant to current drop being worked on. That is because non-running VM could be left over
#	from other drop that was previosly worked on.

CONFIG_COUNT_ONLY_RUNNING_VM=1

#	If set, use the static ip assigned by QTS admin, otherwise use DHCP IP.

DEBUG=1
VM_IPS=""
p2=$2
p1=$1

apt install sshpass -y

if [[ $? -ne 0 ]] ; then
	echo "ERROR. Failed to install sshpass package."
	echo "return code: $?"
	exit 1
fi

#  Count all vms.

TOTAL_VMS=`virsh list --all | grep -i gpu | grep running | wc -l`
echo "TOTAL_VMS: $TOTAL_VMS"

#  Load gim.
#  Start default network.

sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$CONFIG_HOST_IP "modprobe gim"
sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$CONFIG_HOST_IP "virsh net-start default"

#   Set vCPUs to 8.

echo "Starting loop..."
sleep 2

ARR_VM_IP=( )
ARR_VM_NO=()
ARR_VM_NAME=()

function wait_till_ip_read()
{
	for r in ${ARR_VM_NAME[@]} 
	do 
		echo waiting for $r to become pingable...
		for s in {0..50} ; do
			echo issuing virsh domifaddr $r
			tmpIp=`virsh domifaddr $VM_NAME | grep ipv4 | tr -s ' ' | cut -d ' ' -f5 | cut -d '/' -f1`

			ping -q -c 4  $tmpIp
			stat=$?

			if [[ $stat -eq 0 ]] ; then
				echo "Can ping $tmpIp now..."
				break
			fi
			sleep 30
		done

		if [[ $stat -ne 0 ]] ; then
			echo "Error: Can not ping $tmpIp for long time with 10 retries..."
			exit 1	
		fi
	done
}

function print_arrs()
{
	echo --------------------------------------
	echo ${ARR_VM_IP[@]} 
	echo ${ARR_VM_NO[@]} 
	for i in ${ARR_VM_NAME[@]} ; do echo $i; done;
	echo --------------------------------------
}
function clear_arrs()
{
	ARR_VM_IP=( )
	ARR_VM_NO=()
	ARR_VM_NAME=()
}

function get_vm_info()
{
	vmNo=$2
	loopNo=$1
	GPU_INDEX=$vmNo
	VM_INDEX=$(($vmNo+1))
	VM_NAME=`virsh list --all | grep gpu | head -$(($GPU_INDEX+1)) | tail -1  | tr -s ' ' | cut -d ' ' -f3`
	VM_NO=`virsh list --all | grep gpu | head -$(($GPU_INDEX)) | tail -1  | tr -s ' ' | cut -d ' ' -f2`
	VM_IP=`virsh domifaddr $VM_NAME | grep ipv4 | tr -s ' ' | cut -d ' ' -f5 | cut -d '/' -f1`
	ARR_VM_IP+=( $VM_IP ) 
	ARR_VM_NO+=( $VM_NO )
	ARR_VM_NAME+=( $VM_NAME ) 
	DMESG_FILE_NAME=/tmp/dmesg-loop-$loopNo-vm-$vmNo.log
}


echo "Setup memcat on VM-s..."

wait_till_ip_read
clear_arrs
for (( n=0; n < $TOTAL_VMS; n++ ))  ; do
	get_vm_info $i $n
	echo "clear dmesg"
	ssh root@$VM_IP 'rm -rf /tmp/memcat-`hostname`.log'
	ssh root@$VM_IP 'mkdir /memcat'
	scp -r $CONFIG_MEMCAT_SRC_DIR/* root@$VM_IP:/memcat/
	ssh root@$VM_IP 'dpkg -i /memcat/grtev4-x86-runtimes_1.0-145370904_amd64.deb'
	ssh root@$VM_IP 'chmod 755 /memcat/*'
	echo "grtev4-x86-runtimes_1.0-145370904_amd64.deb installation status: $?"
	sleep 3
done

if [[ ! -z $1  ]] ; then
	CONIG_LOOP_TEST_NO=$1
	echo "CONIG_LOOP_TEST_NO is set to $CONIG_LOOP_TEST_NO..."
fi

sleep 3

for (( i=0; i < $CONIG_LOOP_TEST_NO; i++)) ; do
	echo "Loop No. $i"

	sleep 1

	clear_arrs
	for (( n=0; n < $TOTAL_VMS; n++ ))  ; do
		get_vm_info $i $n
		echo "clear dmesg"
		ssh root@$VM_IP 'dmesg --clear'
		echo No. of dmesg line after clear: `ssh root@$VM_IP 'dmesg | wc -l'`
		echo "Done."
		echo "1. checking memcat on $VM_IP..."
		ssh root@$VM_IP 'ls -l /memcat/'
	done

	print_arrs 
	sleep 1

	for m in ${ARR_VM_NAME[@]}  ; do
		#get_vm_info $i $n
		echo "Turning off VM_NAME: $m..."

		virsh shutdown $m &
		#ssh root@$VM_IP 'shutdown now'
	done

	sleep 3
	echo "shutdown/stopped all VM-s..."
	echo "virsh list after shutting down all VM-s: "
	virsh list 

	for (( k=0 ; k < 10; k++)) ; do
		stat=`virsh list | grep running | wc -l`
		echo "Total vms still running after shutdown: $stat"
		if [[ $stat -ne 0 ]] ; then
			echo "Waiting more..."
			sleep 10
		else
			echo "Done. All VM-s are off."
			break
		fi
	done

	if [[ $stat -ne 0 ]] ; then
		echo "Error, not all VMs can be shutdown...!!"
		exit 1
	fi
		

	for m in ${ARR_VM_NAME[@]}  ; do
		#get_vm_info $i $n
		echo "Turning on VM_NAME: $m..."
		virsh start $m &
	done

	print_arrs 
	echo "virsh list after starting all VM-s:"
	virsh list
	sleep 10

	wait_till_ip_read

	for (( n=0; n < $TOTAL_VMS; n++ ))  ; do
		get_vm_info $i $n
		VM_IP=`virsh domifaddr $VM_NAME | grep ipv4 | tr -s ' ' | cut -d ' ' -f5 | cut -d '/' -f1`

		TIME=`date +%H-%M-%S`

		# Load guest driver.

		echo "load AMD gpu" 
		ssh root@$VM_IP 'modprobe amdgpu'

		# Run memcat.

		if [[ $CONFIG_SUPPORT_MEMCAT -eq 1 ]] ; then
			echo "memcat directory content on guest..."
			ssh root@$VM_IP 'ls -l /memcat/'
			echo "Running memcat on $VM_IP..."
			ssh root@$VM_IP '/memcat/amd_memcat.stripped --action write --byte 0x55 >> /tmp/memcat-`hostname`.log'
		fi

		# Copy dmesg to host.

		echo "Copy dmesg to host..."
		ssh root@$VM_IP 'dmesg > /tmp/dmesg'
		TEST_DIR=/g-tracker-142266307/$DATE
		mkdir -p $TEST_DIR
		scp -r root@$VM_IP:/tmp/dmesg $TEST_DIR/dmesg-$VM_NAME-$TIME.log
		scp root@$VM_IP:/tmp/memcat*.log /$TEST_DIR/
	done
	
	stat=`egrep -irn "TRN" $TEST_DIR/dmesg*.log | wc -l`
	echo "No. of lines containing TRN pattern: $stat"

	if [[ $stat -ne 0 ]] ; then
		echo "FOUND THE PATTERN TRN IN DMESG..."
		exit 0
	fi
done

clear_arrs
for (( n=0; n < $TOTAL_VMS; n++ ))  ; do
	get_vm_info $i $n
	ssh root@$VM_IP 'mkdir /memcat'
	scp root@$VM_IP:/tmp/memcat*.log /$TEST_DIR/
done
