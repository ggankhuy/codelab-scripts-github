#   Gibraltar 142266307 issue repro helper script.   
#   Make sure the VMs to be tested and operated on are all running, otherwise result
#   is untested and unpredictable.

DOUBLE_BAR="========================================================"
SINGLE_BAR="--------------------------------------------------------"
CONFIG_SUPPORT_MEMCAT=1
CONFIG_REBOOT=0
CONFIG_MEMCAT_SRC_DIR=/root/memcat/
CONFIG_MEMCAT_DST_DIR=/memcat/
CONFIG_USE_DURATION=1
CONFIG_DURATION_HR=72
CONFIG_RUN_MEMCAT_WITHOUT_AMDGPU=0
CONFIG_DURATION_SEC=$((CONFIG_DURATION_HR * 3600))
CONFIG_LOOP_TEST_NO=1
CONFIG_SET_VCPUCOUNT=0
CONFIG_CLEAR_HOST_DMESG_ON_LOOP=1
CONFIG_GET_REL_VF_BACKGROUND=1
SETUP_GAME_VM_CLIENT=setup-game-vm-client.sh
DATE=`date +%Y%m%d-%H-%M-%S`
DEBUG=0
DEBUG_SYSFS=1

source common.sh

#	Following setting requires great diligence from user of this script. When running flag is set 
#	The  TOTAL_VMS will only count the running VM-s. This could be useful to not count non-running VM
#	which is irrelevant to current drop being worked on. That is because non-running VM could be left over
#	from other drop that was previosly worked on.

CONFIG_COUNT_ONLY_RUNNING_VM=1

#	If set, use the static ip assigned by QTS admin, otherwise use DHCP IP.

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

#   Set vCPUs to 8.

if [[ $TOTAL_VMS -eq 0 ]] ; then 
	echo "TOTAL_VMs are zero...Need running VMs to operate this script on..."
	exit 1
fi

if [[ ! -z $1  ]] ; then
	CONIG_LOOP_TEST_NO=$1
	echo "CONIG_LOOP_TEST_NO is set to $CONIG_LOOP_TEST_NO..."
fi

sleep 1
TIME_LOOP_START=`date +%s`

if [[ $CONFIG_USE_DURATION -eq 1 ]] ; then
	echo "Test loop will continue $CONFIG_DURATION_HR hours..."
	sleep 3
fi

if [[ ! -z $p1  ]] ; then
	CONFIG_LOOP_TEST_NO=$p1
	echo "CONFIG_LOOP_TEST_NO is set to $CONFIG_LOOP_TEST_NO..."
else
	echo "p1 is not supplied from cmdline, using default value for CONFIG_LOOP_TEST_NO: $CONFIG_LOOP_TEST_NO"
fi

sleep 3

echo "Host config..."
echo "clear dmesg on host..."
dmesg --clear

echo "Setup memcat on VM-s..."

clear_arrs

for (( n=0; n < $TOTAL_VMS; n++ ))  ; do
	echo $SINGLE_BAR
	get_vm_info $n
	echo "Setup memcat on $n VM: $VM_IP $VM_NAME..."
	echo "Remove memcat log from guest."
	ssh root@$VM_IP 'rm -rf /tmp/memcat-`hostname`.log'
	echo "Create /memcat directory."
	ssh root@$VM_IP 'mkdir /memcat'
	scp -r $CONFIG_MEMCAT_SRC_DIR/* root@$VM_IP:/memcat/
	ssh root@$VM_IP 'dpkg -i /memcat/grtev4-x86-runtimes_1.0-145370904_amd64.deb'
	ssh root@$VM_IP 'chmod 755 /memcat/*'
	echo "grtev4-x86-runtimes_1.0-145370904_amd64.deb installation status: $?"
	echo "DONE..."
	echo $SINGLE_BAR
done

echo "Starting loop..."
sleep 2
get_bdf
print_arrs 

counter=0

for (( i=0; i < $CONFIG_LOOP_TEST_NO; i++)) ; do
	echo $DOUBLE_BAR
	echo "Loop No. $counter"
	echo $DOUBLE_BAR

	sleep 1
	clear_arrs

	for (( n=0; n < $TOTAL_VMS; n++ ))  ; do
		get_vm_info $n
		echo "clear dmesg"
		ssh root@$VM_IP 'dmesg --clear'
		echo No. of dmesg line after clear: `ssh root@$VM_IP 'dmesg | wc -l'`
		echo "Done."

		if [[ $DEBUG -eq 1 ]] ; then echo "1. checking memcat on $VM_IP..." ;ssh root@$VM_IP 'ls -l /memcat/' ; fi;
	done

	get_bdf

	sleep 1

	for m in ${ARR_VM_NAME[@]}  ; do
		if [[ $CONFIG_REBOOT -eq 1 ]] ; then
			echo "Rebooting VM_NAME $m..."
			virsh reboot $m &
		else
			echo "Turning off VM_NAME: $m..."
			virsh shutdown $m &
		fi
	done

	if [[ $CONFIG_REBOOT -ne 1 ]] ; then
		sleep 10
		echo "shutdown/stopped all VM-s..."
		echo "virsh list after shutting down all VM-s: "
		virsh list 
	
		for m in ${ARR_VM_NAME[@]} 
		do
			for (( k=0 ; k < 10; k++)) 
			do
				stat=`virsh list --all | grep $m | grep "shut off" | wc -l`
				stat1=`virsh list --all | grep $m | grep "shut off" | wc -l`
				echo $stat1
				echo "VM: $m running status: $stat"
				if [[ $stat -ne 1 ]] ; then
					echo "Waiting more..."
					sleep 10
				else
					echo "VM $m has shut off. Moving on..."
					break
				fi
			done
	
			if [[ $stat -ne 1 ]] ; then
				echo "Error, VM $m can not be shutdown...!!"
				virsh list 
				exit 1
			fi
		done

		# relvf calls on all VM.

		counter1=0

		for (( counter1=0; counter1 < $TOTAL_VMS; counter1++ ))  ; do
			if [[ $DEBUG -eq 1 ]] || [[ $DEBUG_SYSFS -eq 1 ]] ; then echo "0000:${ARR_VM_VF[$counter1]} to /sys/bus/pci/devices/0000:${ARR_VM_PF[$counter1]}/relvf" ; fi ;
		
			if [[ $CONFIG_GET_REL_VF_BACKGROUND -eq 0 ]] ; then
				echo 0000:${ARR_VM_VF[$counter1]} > /sys/bus/pci/devices/0000:${ARR_VM_PF[$counter1]}/relvf 
			else
				echo 0000:${ARR_VM_VF[$counter1]} > /sys/bus/pci/devices/0000:${ARR_VM_PF[$counter1]}/relvf  &
			fi
		done 

		counter1=0

		for  counter1 in {0..10} ; do 
			echo "Waiting for all relvf process to finihs..."
			stat=`ps -ax | grep relvf | grep -v grep | grep -v Done | wc -l`
			stat1=`ps -ax | grep relvf | grep -v grep | grep -v Done`

			if [[ $stat -ne 0 ]] ; then
				echo "$counter1..."
				sleep 5
			else
				echo "All relvf processes are finished..."
				break
			fi
		done

		if [[ $counter1 -eq  10 ]] ; then 
			echo "Not all relvf process finished, timeout?..."
			echo $stat1
			exit 1
		fi

		# Issue a hot-reset.

		for (( counter1=0; counter1 < $TOTAL_VMS; counter1++ ))  ; do
			if [[ $DEBUG -eq 1 ]] || [[ $DEBUG_SYSFS -eq 1 ]] ; then echo "0000:${ARR_VM_VF[$counter1]} to /sys/bus/pci/devices/0000:${ARR_VM_PF[$counter1]}/relvf" ; fi ;
		
			if [[ $CONFIG_GET_REL_VF_BACKGROUND -eq 0 ]] ; then
				echo 1 > /sys/bus/pci/devices/0000:${ARR_VM_PF[$counter1]}/hot_reset
			else
				echo 1 > /sys/bus/pci/devices/0000:${ARR_VM_PF[$counter1]}/hot_reset  &
			fi
		done 

		# getvf calls on all VM.

		counter1=0

		for (( counter1=0; counter1 < $TOTAL_VMS; counter1++ ))  ; do
			if [[ $DEBUG -eq 1 ]] || [[ $DEBUG_SYSFS -eq 1 ]]; then echo "32 2048 1 to /sys/bus/pci/devices/0000:${ARR_VM_PF[$counter1]}/getvf" ; fi ; 

			if [[ $CONFIG_GET_REL_VF_BACKGROUND -eq 0 ]] ; then
				echo 32 2048 1 > /sys/bus/pci/devices/0000:${ARR_VM_PF[$counter1]}/getvf
			else
				echo 32 2048 1 > /sys/bus/pci/devices/0000:${ARR_VM_PF[$counter1]}/getvf &
			fi
		done

		for  counter1 in {0..10} ; do 
			echo "Waiting for all getvf process to finihs..."
			stat=`ps -ax | grep getvf | grep -v grep | grep -v Done | wc -l`
			stat1=`ps -ax | grep getvf | grep -v grep | grep -v Done`

			if [[ $stat -ne 0 ]] ; then
				echo "$counter1..."
				sleep 2
			else
				echo "All getvf processes are finished..."
				break
			fi
		done

		if [[ $counter1 -eq  10 ]] ; then 
			echo "Not all getvf process finished, timeout?..."
			echo $stat1
			exit 1
		fi
	
		for m in ${ARR_VM_NAME[@]} ; do
			echo "Turning on VM_NAME: $m..."
			virsh start $m &
		done

		print_arrs 
		echo "virsh list after starting all VM-s:"
	fi

	virsh list
	wait_till_ips_read
	sleep 5

	TIME=`date +%H-%M-%S`

	# Run memcat without loading guest driver, deliberate error. 

	if [[ $CONFIG_RUN_MEMCAT_WITHOUT_AMDGPU  -eq 1 ]] ; then
		for (( n=0; n < $TOTAL_VMS; n++ ))  ; do
			if [[ $CONFIG_SUPPORT_MEMCAT -eq 1 ]] ; then
				if [[ $DEBUG -eq 1 ]] ; then echo "memcat directory content on guest..." ; ssh root@${ARR_VM_IP[$n]} 'ls -l /memcat/' ; fi ;
				if [[ $DEBUG -eq 1 ]] ; then echo "Running memcat on ${ARR_VM_IP[$n]}..." ; fi ;
				ssh root@${ARR_VM_IP[$n]} "for i in {0..10}; do /memcat/amd_memcat.stripped --action write --byte 0x55 >> /tmp/memcat-${ARR_VM_NAME[$n]}-loop-$n.log ; done"
			fi
		done
	fi

	# Load guest driver.
	
	for (( n=0; n < $TOTAL_VMS; n++ ))  ; do
		echo "load AMD gpu..." 
		ssh root@${ARR_VM_IP[$n]} 'modprobe amdgpu' &
	done

	# Run memcat.

	for (( n=0; n < $TOTAL_VMS; n++ ))  ; do
		if [[ $CONFIG_SUPPORT_MEMCAT -eq 1 ]] ; then

			if [[ $DEBUG -eq 1 ]] ; then echo "memcat directory content on guest..." ; ssh root@${ARR_VM_IP[$n]} 'ls -l /memcat/' ; fi ;
			if [[ $DEBUG -eq 1 ]] ; then echo "Running memcat on ${ARR_VM_IP[$n]}..." ; fi ;
			ssh root@${ARR_VM_IP[$n]} "for i in {0..10}; do /memcat/amd_memcat.stripped --action write --byte 0x55 >> /tmp/memcat-${ARR_VM_NAME[$n]}-loop-$n.log ; done"
		fi
	done

	for (( n=0; n < $TOTAL_VMS; n++ ))  ; do
		echo "unload AMD gpu"... 
		ssh root@${ARR_VM_IP[$n]} 'modprobe -r amdgpu' &
	done
	
	# Run memcat after unload guest driver again, deliberate error.


	sleep 10

	if [[ $CONFIG_RUN_MEMCAT_WITHOUT_AMDGPU -eq 1 ]] ; then
		for (( n=0; n < $TOTAL_VMS; n++ ))  ; do
			if [[ $CONFIG_SUPPORT_MEMCAT -eq 1 ]] ; then
				if [[ $DEBUG -eq 1 ]] ; then echo "memcat directory content on guest..." ; ssh root@${ARR_VM_IP[$n]} 'ls -l /memcat/' ; fi ;
				if [[ $DEBUG -eq 1 ]] ; then echo "Running memcat on ${ARR_VM_IP[$n]}..." ; fi ;
				ssh root@${ARR_VM_IP[$n]} "for i in {0..10}; do /memcat/amd_memcat.stripped --action write --byte 0x55 >> /tmp/memcat-${ARR_VM_NAME[$n]}-loop-$n.log ; done"
			fi
		done
	fi

	sleep 5

	for (( n=0; n < $TOTAL_VMS; n++ ))  ; do
		# Copy dmesg to host.
	
		TIME=`date +%H-%M-%S`
		DMESG_GUEST_DST_FILENAME=dmesg-guest-${ARR_VM_NAME[$n]}-$TIME.log
		echo "Copy dmesg to host... as $DMESG_GUEST_DST_FILENAME"
		ssh root@${ARR_VM_IP[$n]} 'dmesg > /tmp/dmesg'
		TEST_DIR=/g-tracker-142266307/$DATE
		mkdir -p $TEST_DIR
		
		scp root@${ARR_VM_IP[$n]}:/tmp/dmesg $TEST_DIR/$DMESG_GUEST_DST_FILENAME
		scp root@${ARR_VM_IP[$n]}:/tmp/memcat-${ARR_VM_NAME[$n]}-loop-$n.log /$TEST_DIR/
	done
	
	stat=`egrep -irn "TRN" $TEST_DIR/dmesg*.log | wc -l`
	echo "No. of lines containing TRN pattern: $stat"

	if [[ $stat -ne 0 ]] ; then
		echo "FOUND THE PATTERN TRN IN DMESG..."
		exit 0
	fi

	if [[ $CONFIG_USE_DURATION -eq 1 ]] ; then
		i=$((i-1))
		echo "loop variable i: $i"
		TIME_LOOP_CURRENT=`date +%s`
		loopDurationSec=$((TIME_LOOP_CURRENT-TIME_LOOP_START))
		loopDurationHr=$((loopDurationSec/3600))
		loopDurationMin=$((loopDurationSec/60))
		echo "Test run duration: $loopDurationMin minutes..."	

		if [[ $loopDurationSec -gt $CONFIG_DURATION_SEC ]] ; then
			echo "End of test loop.  Test ran for $loopDurationSec seconds, or $loopDurationHr hours..."
			break
		fi
	fi

	if [[ $CONFIG_CLEAR_HOST_DMESG_ON_LOOP -eq 1 ]] ; then
		dmesg  > /$TEST_DIR/dmesg-host-loop-$counter.log
		dmesg --clear
	fi
	

	counter=$((counter+1))
done

if [[ $CONFIG_CLEAR_HOST_DMESG_ON_LOOP -ne 1 ]] ; then
	dmesg  > /$TEST_DIR/dmesg-host.log
fi

clear_arrs
