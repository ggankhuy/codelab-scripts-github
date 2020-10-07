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

# POWER CYCLE TYPE

CONFIG_PC_REBOOT=1

#	power off and on. Interval between off and on is dictated by CONFIG_PC_POWERCYCLE_IN in seconds.

CONFIG_PC_POWERCYCLE=2 
CONFIG_PC_POWERCYCLE_INTERNAL=1

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
	lspci | grep -i amd | grep Disp` >  $CONFIG_PATH_LOG/lspci.log
	$CONFIG_PATH_AMDVBFLASH -i > $CONFIG_PATH_LOG/amdvbflash.log
	echo "iteration $i: " >> $DIRNAME/summary.log
	echo No. of gpu-s detected by lspci: `lspci | grep -i amd | grep Disp | wc -l` >> $DIRNAME/summary.log
	echo No. of gpu-s detected by amdvbflash: `$CONFIG_PATH_AMDVBFLASH` >> $DIRNAME/summary.log
	for (( j=$CONFIG_GPU_FLASH_IDX_MIN ; j < $CONFIG_GPU_FLASH_IDX_MAX; j++ )) ; do
		echo "Flashing gpu $j..."
		$$CONFIG_PATH_AMDVBFLASH -p $j $CONFIG_PATH_VBIOS
	done

	echo "Powercycling..."

	if [[ $CONFIG_PC_TYPE==$CONFIG_PC_REBOOT ]] ; then
		"echo Powercycle type: reboot..."
		reboot
	elif [[ $CONFIG_PC_TYPE=$CONFIG_PC_POWERCYCLE ]] ; then
		"echo Powercycle type: power off and on..."
		
	fi
done




