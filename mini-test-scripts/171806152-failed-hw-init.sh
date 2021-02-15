#Synopsys:
# - for loop time (CONFIG_ITER_POWERCYCLE)
# - - Powercycle or reboot (powercycle is only for ST only)
# - - Once powercycle/reboot complete, load and unload GIM N-times (CONFIG_ITER_GIM_RELOAD)

DATE=`date +%Y%m%d-%H-%M-%S`
DIRNAME=162454034-result/$DATE/
mkdir -p $DIRNAME

# GPU flash range
CONFIG_GPU_FLASH_IDX_MIN=2
CONFIG_GPU_FLASH_IDX_MAX=12
#CONFIG_GPU_FLASH_IDX_MAX=5

# amdvbflash path
CONFIG_PATH_AMDVBFLASH=/root/tools/amdvbflash/amdvbflash-4.74/amdvbflash
CONFIG_PATH_VBIOS=/drop/20200918/linux_host_package/vbios/NAVI12/Gemini/D3020100.101

#  BMC access

CONFIG_BMC_IP="10.216.52.241"
CONFIG_BMC_USERNAME=root
CONFIG_BMC_PW=0penBmc

# host ip access

CONFIG_OS_IP="10.216.52.232"
CONFIG_OS_USERNAME=root
CONFIG_OS_PW=amd1234
CONFIG_PING_TIMEOUT=300
CONFIG_PING_INTERVAL=15
EXTRA_SLEEP_SECONDS=60 # No. of seconds to sleep after ping is detected, to give O/S to boot complete booting.
# POWER CYCLE TYPE

CONFIG_PC_REBOOT=1

#	power off and on. Interval between off and on is dictated by CONFIG_PC_POWERCYCLE_IN in seconds.

CONFIG_PC_POWERCYCLE=2 
CONFIG_PC_POWERCYCLE_INTERVAL=30

#	Powercycle or reboot.

CONFIG_PC_TYPE=$CONFIG_PC_REBOOT

#	number of test to repeat

CONFIG_ITER_POWERCYCLE=100 
CONFIG_ITER_GIM_RELOAD=5

CONFIG_DEBUG_ENABLE_POWERCYCLE=1

#	Whether clear cmd enabled, friendly for terminal but not for ssh log and vice versa.
CONFIG_ENABLE_CLEAR=0

for var in "$@"
do
    if [[ ! -z `echo "$var" | grep "help"` ]]  ; then
	if [[ $CONFIG_ENABLE_CLEAR -eq 1 ]] ; then clear ; fi
	echo "-----------------------------"
        echo "$0 iter=<number of loops> bmcip=<bmc ip> osip=<OS IP>"
	echo "-----------------------------"
	exit 0
    fi

    if [[ ! -z `echo "$var" | grep "iter="` ]]  ; then
        echo "number of loops: $var"
        CONFIG_ITER_POWERCYCLE=`echo $var | cut -d '=' -f2`
        echo "CONFIG_ITER_POWERCYCLE: $CONFIG_ITER_POWERCYCLE"
    fi

    if [[ ! -z `echo "$var" | grep "bmcip="` ]]  ; then
        CONFIG_BMC_IP=`echo $var | cut -d '=' -f2`
        echo "CONFIG_BMC_IP: $CONFIG_BMC_IP"

        if [[ $CONFIG_PC_REBOOT -eq 1 ]] ; then
            echo "Reboot specified, bmc ping is bypassed."
        else
        	ping -c 2 $CONFIG_BMC_IP
	        if [[ $? -ne 0 ]] ; then echo "$CONFIG_BMC_IP is not pingable..."; exit 1 ; fi            
        fi
    fi

    if [[ ! -z `echo "$var" | grep "osip="` ]]  ; then
        CONFIG_OS_IP=`echo $var | cut -d '=' -f2`
        echo "CONFIG_OS_IP: $CONFIG_OS_IP"
	if [[ $? -ne 0 ]] ; then "$CONFIG_OS_IP is not pingable..."; exit 1 ; fi
    fi
done

for (( i=0 ; i < $CONFIG_ITER_POWERCYCLE ; i++ )) ; do
	CONFIG_PATH_LOG=$DIRNAME/$i/
	mkdir -p $CONFIG_PATH_LOG
	echo "Loop $i..."
	sshpass -p $CONFIG_OS_PW ssh -o StrictHostKeyChecking=no $CONFIG_OS_USERNAME@$CONFIG_OS_IP "lspci | grep -i amd | grep Disp" > $CONFIG_PATH_LOG/lspci.log
	sshpass -p $CONFIG_OS_PW ssh -o StrictHostKeyChecking=no $CONFIG_OS_USERNAME@$CONFIG_OS_IP "$CONFIG_PATH_AMDVBFLASH -i" >  $CONFIG_PATH_LOG/amdbvflash.log
	echo "--- iteration $i: ---" >> $DIRNAME/summary.log
	res=`sshpass -p $CONFIG_OS_PW ssh -o StrictHostKeyChecking=no $CONFIG_OS_USERNAME@$CONFIG_OS_IP "lspci | grep -i amd | grep Disp | wc -l" | tr -d '\n'`
	echo "No. of gpu-s detected by lspci : $res " 
	echo "No. of gpu-s detected by lspci : $res " | tee -a $DIRNAME/summary.log

	if [[ $CONFIG_DEBUG_ENABLE_POWERCYCLE -ne 0 ]] ; then
		echo "Powercycling..." 
	
		if [[ $CONFIG_PC_TYPE -eq $CONFIG_PC_REBOOT ]] ; then
			echo "Powercycle type: reboot..." | tee -a  >> $DIRNAME/summary.log
			sshpass -p $CONFIG_OS_PW ssh -o StrictHostKeyChecking=no $CONFIG_OS_USERNAME@$CONFIG_OS_IP "reboot"
			sleep 30
		elif [[ $CONFIG_PC_TYPE -eq $CONFIG_PC_POWERCYCLE ]] ; then
			echo "Powercycle type: power off and on..." | tee -a  >> $DIRNAME/summary.log
			sshpass -p $CONFIG_BMC_PW ssh -o StrictHostKeyChecking=no $CONFIG_BMC_USERNAME@$CONFIG_BMC_IP "python /home/root/BMC_Scripts/shut_down_system.py"
			sleep 10
			sleep $CONFIG_PC_POWERCYCLE_INTERVAL
			sshpass -p $CONFIG_BMC_PW ssh -o StrictHostKeyChecking=no $CONFIG_BMC_USERNAME@$CONFIG_BMC_IP "python /home/root/BMC_Scripts/boot_up_system.py"	
		else
			echo "Can not recognize powercycle type: $CONFIG_PC_TYPE"
			exit 1
		fi
	
		CONFIG_PING_TIMEOUT_ERROR=1
		for (( k = 0 ; k < $CONFIG_PING_TIMEOUT ; k+=$CONFIG_PING_INTERVAL )) ; do
			ping -c 2 $CONFIG_OS_IP
			ping_stat=$?
	
			if [[ $ping_stat -eq 0 ]] ; then
				echo "Can ping OS now, give it a $EXTRA_SLEEP_SECONDS seconds..."
				sleep $EXTRA_SLEEP_SECONDS
				CONFIG_PING_TIMEOUT_ERROR=0
				break
			fi
			sleep $CONFIG_PING_INTERVAL
		done
	
		if [[ $CONFIG_PING_TIMEOUT_ERROR -eq 1 ]] ; then
			echo "Timeout trying to ping OS for $CONFIG_PING_TIMEOUT seconds, giving up..."
			exit 1
		fi
	fi

    # Reload gim N times within powercycle.
   	sshpass -p $CONFIG_OS_PW ssh -o StrictHostKeyChecking=no $CONFIG_OS_USERNAME@$CONFIG_OS_IP "dmesg" > $CONFIG_PATH_LOG/dmesg.after.reboot.$i.log

    for (( j=0 ; j < $CONFIG_ITER_GIM_RELOAD ; j++ )) ; do
    	sshpass -p $CONFIG_OS_PW ssh -o StrictHostKeyChecking=no $CONFIG_OS_USERNAME@$CONFIG_OS_IP "dmesg --clear ; modprobe gim ; dmesg" > $CONFIG_PATH_LOG/dmesg.after.gim.load.$i.$j.log
    	sshpass -p $CONFIG_OS_PW ssh -o StrictHostKeyChecking=no $CONFIG_OS_USERNAME@$CONFIG_OS_IP "dmesg --clear ; modprobe -r gim ; dmesg" > $CONFIG_PATH_LOG/dmesg.after.gim.unload.$i.$j.log
    done
done
