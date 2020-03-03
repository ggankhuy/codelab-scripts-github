#   Gibraltar 142266307 issue repro helper script.   
#   Make sure the VMs to be tested and operated on are all running, otherwise result
#   is untested and unpredictable.

DOUBLE_BAR="========================================================"
SINGLE_BAR="--------------------------------------------------------"
CONFIG_SUPPORT_MEMCAT=1
CONFIG_REBOOT=1
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

function dmesg_check_psp_bl_stat()
{
  vm_name=$1

  test_report_folder=./test-report/$0/dmesg
  mkdir -p $test_report_folder
  touch $test_report_folder/dmesg-`date +%Y%m%d-%H-%M-%S`.log
  touch $test_report_folder/dmesg-all.log
  dmesg > $test_report_folder/dmesg-`date +%Y%m%d-%H-%M-%S`.log
  dmesg >> $test_report_folder/dmesg-all.log
  ls -l $test_report_folder
  sleep 3

  if [[ `dmesg | grep "mmMP0_SMN_C2PMSG_33 reads non-zero value"` ]] ; then
        echo "mmMP0_SMN_C2PMSG_33 read none zero value" 
        exit 1
  elif [[ `dmesg | grep "mmMP0_SMN_C2PMSG_33 read OK."` ]] ; then
        echo "mmMP0_SMN_C2PMSG_33 reads zero value, OK."
  else
        echo "dmesg contains no mmMP0_SMN_C2PMSG_33 status, logic error, leaving."
        exit 1
  fi
}

#	Following setting requires great diligence from user of this script. When running flag is set 
#	The  TOTAL_VMS will only count the running VM-s. This could be useful to not count non-running VM
#	which is irrelevant to current drop being worked on. That is because non-running VM could be left over
#	from other drop that was previosly worked on.

CONFIG_COUNT_ONLY_RUNNING_VM=1

#	If set, use the static ip assigned by QTS admin, otherwise use DHCP IP.

VM_IPS=""
p2=$2
p1=$1

#apt install sshpass -y

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

#   start with VM-s running

#   loop
#   - clear dmesg
#   - shutdown VM-s
#   - unload GIM
#   - load GIM 
#   - start VM-s
#   - (optional) load amdgpus on guest vm-s.
#   - check psp_bl from dmesg
#       - exit if Rx is 80000000.

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

#   - clear dmesg

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

#   - shutdown VM-s

	for m in ${ARR_VM_NAME[@]}  ; do
		echo "Turning off VM_NAME: $m..."
		virsh shutdown $m
	done

#   - unload GIM

    modprobe -r gim

    echo "GIM status using lsmod after unload..."
    if [[ -z `lsmod | grep gim` ]] ; then
        echo "GIM unload is ok."
    else
        echo "GIM unload appears unsuccessful."
        lsmod | grep gim
    fi
    sleep 5

#   - load GIM 

    modprobe gim

#   - start VM-s

	for m in ${ARR_VM_NAME[@]}  ; do
		echo "Turning on VM_NAME: $m..."
		virsh start $m 
	done

#   - check psp_bl from dmesg
#       - exit if Rx is 80000000.

    dmesg_check_psp_bl_stat

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
done

exit 0

