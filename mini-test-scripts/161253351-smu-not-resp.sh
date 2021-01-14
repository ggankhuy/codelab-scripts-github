# Alternatively boot between legacy and libgv gim version between powercycle.
# Operational briefs are as follows:
# 1. For every other boot build and install libgv and legacy-gim on other boots.
# 2. For libgv flash latest libgv compatible VBIOS (i.e. currently D04), for legacy-gim, flash load Y03 VBIOS.
# 3. Powercycle (for G servers) or reboot (for non-G servers)
# 3. Load gim  (libgv or legacy-gim)
# 4. Start all VM-s

DATE=`date +%Y%m%d-%H-%M-%S`
DIRNAME=161253351-result/$DATE/
mkdir -p $DIRNAME

# 	MKM ST 47 configuration

HOST_IP="10.6.168.76"
BMC_IP="10.6.170.182"
VBIOS_5438=/drop/20201023/linux_host_package/vbios/V340L/D0531800.D04
VBIOS_415=/drop/drop-2019-q3-rc8-GOOD-install/drop-2019-q3-rc8/vbios/V340L/D0531800.Y03

#	ST SEMI02 configuration

HOST_IP="10.216.52.50"
BMC_IP="10.216.52.51"
VBIOS_5438=/drop/20201023/linux_host_package/vbios/NAVI12/Gemini
VBIOS_415=$VBIOS_5438

#	IXT39 configuration

HOST_IP="10.216.66.54"
VBIOS_5438=/drop/20201023/linux_host_package/vbios/V340L/D0531800.D04
VBIOS_415=/drop/drop-2019-q3-rc8-GOOD-install/drop-2019-q3-rc8/vbios/V340L/D0531800.Y03

#	misc. configuration 

LOOP_COUNT=3
EXTRA_SLEEP=14
AMDVBFLASH_PATH=/root/tools/amdvbflash/amdvbflash-4.68/amdvbflash

BMC_PW=0penBmc
BMC_USER=root
HOST_PW=amd1234
HOST_USER=root

HOST_RESPONSIVE=0
CONFIG_DISABLE_FLASH_VBIOS=0
CONFIG_DISABLE_HOST_DRV_BUILD=0

#	Reboot the server instead of powercycle. Powercycle is only supported on ST or other G servers.
#	If you set the CONFIG_REBOOT=0 and if it is not G server, result is not predictable.
#	Powercycle is preferable test option over reboot. 

CONFIG_REBOOT=1

function build_install_legacy_gim() {
	if [[ $CONFIG_DISABLE_HOST_DRV_BUILD -ne 1 ]] ; then
		cmds=( "mkdir /git.co/" "cd /git.co/ ; git clone https://ggghamd:amd1234A%23@github.com/ggghamd/cp-Gibraltar-GIM.git"  \
			"cd /git.co/cp-Gibraltar-GIM; ./dkms.sh gim 1.0" "modprobe gim")
		for (( i=0; i < ${#cmds[@]}; i++ )); do
			echo "cmd: ${cmds[$i]}" ; sleep 1
			sshpass -p $HOST_PW ssh -o StrictHostKeyChecking=no $HOST_USER@$HOST_IP "${cmds[$i]}"
		done
	else
		echo "Bypassing host driver build..."
	fi
}

function build_install_libgv() {
	if [[ $CONFIG_DISABLE_HOST_DRV_BUILD -ne 1 ]] ; then
		cmds=( "mkdir /git.co/" "cd /git.co/ ; git clone https://ggghamd:amd1234A%23@github.com/ggghamd/ad-hoc-scripts.git"  \
			"cd /git.co/ad-hoc-scripts ; ./dkms.sh gim 2.0.1.G.20201023" "modprobe gim" "popd" )
		for (( i=0; i < ${#cmds[@]}; i++ )); do
			echo "cmd: ${cmds[$i]}" ; sleep 1
			sshpass -p $HOST_PW ssh -o StrictHostKeyChecking=no $HOST_USER@$HOST_IP "${cmds[$i]}"
		done
	else
		echo "Bypassing host driver build..."
	fi
}

#	Power cycle the ST server. (Semitruck). Attempting this on non google server has unpredictable result. 
#	If host is not responding after 3 tries, function will exit to terminal with exit code 1.

function powercycle_server()
{
	HOST_RESPONSIVE=0

	if [[ $CONFIG_REBOOT -eq 1 ]] ; then
		echo "CONFIG_REBOOT is enabled. Rebooting instead of powercycling..." ; sleep 5
		sshpass -p $HOST_PW ssh -o StrictHostKeyChecking=no $HOST_USER@$HOST_IP "reboot"
		sleep $EXTRA_SLEEP
		for i in {0..60} ;
		do
			ping -c 1 $HOST_IP 
			if [[ $? -eq 0 ]] ; then
				echo "system is pingable, give another $EXTRA_SLEEP sec..."
				sleep $EXTRA_SLEEP
				HOST_RESPONSIVE=1
				break				
			fi
			sleep 5
		done	
	else
		# Power cycle the server now.

		for j in {0..3} ; 
		do
			echo "Trying to power cycle...Loop No. $j"
			sleep $EXTRA_SLEEP
			sshpass -p $BMC_PW ssh -o StrictHostKeyChecking=no $BMC_USER@$BMC_IP 'cd ~/BMC_Scripts/ ; python shut_down_system.py ; sleep 15; python boot_up_system.py'
			sleep $EXTRA_SLEEP
			for i in {0..60} ;
			do
				ping -c 1 $HOST_IP 
				if [[ $? -eq 0 ]] ; then
					echo "system is pingable, give another $EXTRA_SLEEP sec..."
					sleep $EXTRA_SLEEP
					HOST_RESPONSIVE=1
					break				
				fi
				sleep 5
			done	
		done
	fi
	if [[ $HOST_RESPONSIVE -ne 1 ]] ; then echo "Host is not responding after N? retries to powercycle. Can not continue..." ; exit 1; fi
}
 
for loopCnt in {0..200};
do
	echo "--- LOOP COUNT $loopCnt ----" | tee -a $DIRNAME/summary.log

	echo "setting kernel 4.15... and vbios to $VBIOS_415"
	for m in {0..3}
	do
		if [[ $CONFIG_DISABLE_FLASH_VBIOS -eq 1 ]] ; then
			sleep 1
			echo sshpass -p $HOST_PW ssh -o StrictHostKeyChecking=no $HOST_USER@$HOST_IP "$AMDVBFLASH_PATH -f -p $m $VBIOS_415"
		else
			sshpass -p $HOST_PW ssh -o StrictHostKeyChecking=no $HOST_USER@$HOST_IP "$AMDVBFLASH_PATH -f -p $m $VBIOS_415"
		fi
	done

	sshpass -p $HOST_PW ssh -o StrictHostKeyChecking=no $HOST_USER@$HOST_IP 'for k in {1..4} ; do virsh shutdown vats-test-0$k ; done'
	build_install_legacy_gim
	powercycle_server

	echo "vbios/gim:" 
	sshpass -p $HOST_PW ssh -o StrictHostKeyChecking=no $HOST_USER@$HOST_IP "$AMDVBFLASH_PATH -i" | tee -a $DIRNAME/summary.log
	sshpass -p $HOST_PW ssh -o StrictHostKeyChecking=no $HOST_USER@$HOST_IP "modprobe -r gim ; dmesg --clear ; modprobe gim ; dmesg | grep \"GPU IOV MODULE\"" | tee -a $DIRNAME/summary.log
	sshpass -p $HOST_PW ssh -o StrictHostKeyChecking=no $HOST_USER@$HOST_IP 'virsh net-start default ; dmesg --clear ; for k in {1..4} ; do echo $k ; virsh start vats-test-0$k ; done ; dmesg' | tee -a $DIRNAME/$loopCnt.log

	for m in {0..3}
	do
		if [[ $CONFIG_DISABLE_FLASH_VBIOS -eq 1 ]] ; then
			sleep 1
			echo sshpass -p $HOST_PW ssh -o StrictHostKeyChecking=no $HOST_USER@$HOST_IP "$AMDVBFLASH_PATH -f -p $m $VBIOS_5438"
		else
			sshpass -p $HOST_PW ssh -o StrictHostKeyChecking=no $HOST_USER@$HOST_IP "$AMDVBFLASH_PATH -f -p $m $VBIOS_5438"
		fi
	done

	sshpass -p $HOST_PW ssh -o StrictHostKeyChecking=no $HOST_USER@$HOST_IP 'for k in {1..4} ; do virsh shutdown vats-test-0$k ; done'
	build_install_libgv
	powercycle_server

	echo "vbios/gim:" 
	sshpass -p $HOST_PW ssh -o StrictHostKeyChecking=no $HOST_USER@$HOST_IP "$AMDVBFLASH_PATH -i" | tee -a $DIRNAME/summary.log
	sshpass -p $HOST_PW ssh -o StrictHostKeyChecking=no $HOST_USER@$HOST_IP "modprobe -r gim ; dmesg --clear ; modprobe gim ; dmesg | grep \"GPU IOV MODULE\"" | tee -a $DIRNAME/summary.log
	sshpass -p $HOST_PW ssh -o StrictHostKeyChecking=no $HOST_USER@$HOST_IP 'virsh net-start default; dmesg --clear ; for k in {1..4} ; do virsh start vats-test-0$k ; done ; dmesg' | tee -a $DIRNAME/$loopCnt.log
done	
