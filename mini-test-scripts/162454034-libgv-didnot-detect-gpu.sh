# The script only works for semitrucks where bmc is accessible through ssh.

DATE=`date +%Y%m%d-%H-%M-%S`
DIRNAME=162454034-result/$DATE/
mkdir -p $DIRNAME

# GPU flash range
CONFIG_GPU_FLASH_IDX_MIN=2
CONFIG_GPU_FLASH_IDX_MAX=13

# amdvbflash path
CONFIG_PATH_AMDVBFLASH=/root/tools/amdvbflash/amdvbflash-4.74
CONFIG_PATH_VBIOS=/drop/20200918/linux_host_package/vbios/NAVI12/Gemini/D3020100.101

#  BMC access

CONFIG_BMC_IP="10.216.52.241"
CONFIG_BMC_USERNAME=root
CONFIG_BMC_PW=OpenBmc

# host ip access

CONFIG_OS_IP="10.216.52.232"
CONFIG_OS_USERNAME=root
COFNIG_OS_PW=amd1234
CONFIG_PING_TIMEOUT=300
CONFIG_PING_INTERVAL=15

# POWER CYCLE TYPE

CONFIG_PC_REBOOT=1

#	power off and on. Interval between off and on is dictated by CONFIG_PC_POWERCYCLE_IN in seconds.

CONFIG_PC_POWERCYCLE=2 
CONFIG_PC_POWERCYCLE_INTERVAL=30

#	Powercycle or reboot.

CONFIG_PC_TYPE=$CONFIG_PC_REBOOT

#	number of test to repeat

CONFIG_ITER=100 

for var in "$@"
do
    if [[ ! -z `echo "$var" | grep "help"` ]]  ; then
	clear
	echo "-----------------------------"
        echo "$0 iter=<number of loops> bmcip=<bmc ip> osip=<OS IP>"
	echo "-----------------------------"
	exit 0
    fi

    if [[ ! -z `echo "$var" | grep "iter="` ]]  ; then
        echo "number of loops: $var"
        CONFIG_ITER=`echo $var | cut -d '=' -f2`
        echo "CONFIG_ITER: $CONFIG_ITER"
    fi

    if [[ ! -z `echo "$var" | grep "bmcip="` ]]  ; then
        CONFIG_BMC_IP=`echo $var | cut -d '=' -f2`
        echo "CONFIG_BMC_IP: $CONFIG_BMC_IP"
    fi
    if [[ ! -z `echo "$var" | grep "osip="` ]]  ; then
        CONFIG_OS_IP=`echo $var | cut -d '=' -f2`
        echo "CONFIG_OS_IP: $CONFIG_OS_IP"
    fi
done

for (( i=0 ; i < $CONFIG_ITER ; i++ )) ; do
	CONFIG_PATH_LOG=$DIRNAME/$i/
	sshpass -p $CONFIG_OS_PW ssh -o StrictHostKeyChecking=no $CONFIG_OS_USERNAME@$CONFIG_OS_IP "lspci | grep -i amd | grep Disp" >  $CONFIG_PATH_LOG/lspci.log
	sshpass -p $CONFIG_OS_PW ssh -o StrictHostKeyChecking=no $CONFIG_OS_USERNAME@$CONFIG_OS_IP "$CONFIG_PATH_AMDVBFLASH -i" >  $CONFIG_PATH_LOG/amdbvflash.log
	echo "iteration $i: " >> $DIRNAME/summary.log
	echo No. of gpu-s detected by lspci: "
	sshpass -p $CONFIG_OS_PW ssh -o StrictHostKeyChecking=no $CONFIG_OS_USERNAME@$CONFIG_OS_IP "lspci | grep -i amd | grep Disp | wc -l" >> $DIRNAME/summary.log
	echo No. of gpu-s detected by amdvbflash: 
	sshpass -p $CONFIG_OS_PW ssh -o StrictHostKeyChecking=no $CONFIG_OS_USERNAME@$CONFIG_OS_IP "$CONFIG_PATH_AMDVBFLASH -i" >> $DIRNAME/summary.log

	for (( j=$CONFIG_GPU_FLASH_IDX_MIN ; j < $CONFIG_GPU_FLASH_IDX_MAX; j++ )) ; do
		echo "Flashing gpu $j..."
		sshpass -p $CONFIG_OS_PW ssh -o StrictHostKeyChecking=no $CONFIG_OS_USERNAME@$CONFIG_OS_IP "$CONFIG_PATH_AMDVBFLASH -p $j $CONFIG_PATH_VBIO"
	done

	echo "Powercycling..."

	if [[ $CONFIG_PC_TYPE==$CONFIG_PC_REBOOT ]] ; then
		"echo Powercycle type: reboot..."
		sshpass -p $CONFIG_OS_PW ssh -o StrictHostKeyChecking=no $CONFIG_OS_USERNAME@$CONFIG_OS_IP "reboot"
	elif [[ $CONFIG_PC_TYPE=$CONFIG_PC_POWERCYCLE ]] ; then
		"echo Powercycle type: power off and on..."
		sshpass -p $CONFIG_BMC_PW ssh -o StrictHostKeyChecking=no $CONFIG_BMC_USERNAME@$CONFIG_BMC_IP "python /root/BMC_Scripts/shut_down_system.py"
		sleep $CONFIG_PC_POWERCYCLE_INTERVAL
		sshpass -p $CONFIG_BMC_PW ssh -o StrictHostKeyChecking=no $CONFIG_BMC_USERNAME@$CONFIG_BMC_IP "python /root/BMC_Scripts/boot_up_system.py"	
	fi

	CONFIG_PING_TIMEOUT_ERROR=1
	for (( k = 0 ; k < $CONFIG_PING_TIMEOUT ; k+=$CONFIG_PING_INTERVAL )) ; do
		ping -c 2 $CONFIG_OS_IP
		ping_stat=$?

		if [[ $ping_stat -eq 0 ]] ; then
			echo "Can ping OS now..."
			CONFIG_PING_TIMEOUT_ERROR=0
			break
		fi
		sleep $CONFIG_PING_INTERVAL
	done

	if [[ $CONFIG_PING_TIMEOUT_ERROR -eq 1 ]] ; then
		echo "Timeout trying to ping OS for $CONFIG_PING_TIMEOUT seconds, giving up..."
		exit 1
	fi
done




