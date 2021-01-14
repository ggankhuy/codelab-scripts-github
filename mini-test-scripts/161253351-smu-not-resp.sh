# Alternativelyboot between 5.4.38 and 4.15 kernel. 
# For 5.4.39, flash 20201023 VBIOS
# For 4.15 kernel, flash 19Q3 VBIOS.

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

#	misc. configuration 

LOOP_COUNT=3
EXTRA_SLEEP=14
AMDVBFLASH_PATH=/root/tools/amdvbflash/amdvbflash-4.68/amdvbflash

BMC_PW=0penBmc
BMC_USER=root
HOST_PW=amd1234
HOST_USER=root

HOST_RESPONSIVE=0
CONFIG_FAKE_FLASH_VBIOS=1

#	Reboot the server instead of powercycle. Powercycle is only supported on ST or other G servers.
#	If you set the CONFIG_REBOOT=0 and if it is not G server, result is not predictable.
#	Powercycle is preferable test option over reboot. 

CONFIG_REBOOT=1

#	Power cycle the ST server. (Semitruck). Attempting this on non google server has unpredictable result. 
#	If host is not responding after 3 tries, function will exit to terminal with exit code 1.

function build_install_legacy_gim() {
	mkdir /git.co/ ; pushd /git.co
	#git clone https://ggghamd:amd1234A%23@github.com/AMD-CloudGPU/Gibraltar-GIM
	git clone https://ggghamd:amd1234A%23@github.com/ggghamd/cp-Gibraltar-GIM.git
	cd Gibraltar-GIM
	./dkms.sh gim 1.0
	dmesg --clear
	modprobe gim
	popd
}

function build_install_libgv() {
	mkdir /git.co/ ; pushd /git.co
	#git clone https://ggghamd:amd1234A%23@github.com/AMD-CloudGPU/Gibraltar-GIM
	git clone https://ggghamd:amd1234A%23@github.com/ggghamd/ad-hoc-scripts.git
	popd
	/git.co/ad-hoc-scripts/dkms.sh gim 2.0.1.G.20201023
	dmesg --clear
	modprobe gim
}

function powercycle_st_server()
{
	if [[ $CONFIG_REBOOT -eq 1 ]] ; then
		echo "CONFIG_REBOOT is enabled. Rebooting instead of powercycling..." ; sleep 5
		sshpass -p $HOST_PW ssh -o StrictHostKeyChecking=no $HOST_USER@$HOST_IP "reboot"
		return 
	fi

	echo "Powercycling ..." 

	# Power cycle the server now.

	HOST_RESPONSIVE=0

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
	if [[ $HOST_RESPONSIVE -ne 1 ]] ; then echo "Host is not responding after 3 retries to powercycle. Can not continue..." ; exit 1; fi
}
 
for i in {0..3};
do
	echo "--- LOOP COUNT $i ----"

	echo "setting kernel 4.15... and vbios to $VBIOS_415"
	for i in {0..14}
	do
		if [[ $CONFIG_FAKE_FLASH_VBIOS -eq 1 ]] ; then
			sleep 1
			echo sshpass -p $HOST_PW ssh -o StrictHostKeyChecking=no $HOST_USER@$HOST_IP "$AMDVBFLASH_PATH -f -p $i $VBIOS_415"
			
		else
			#sshpass -p $HOST_PW ssh -o StrictHostKeyChecking=no $HOST_USER@$HOST_IP "$AMDVBFLASH_PATH -f -p $i $VBIOS_415"
			#ret=$?
			#if [[ ! -z $ret ]] ; then echo "ret: $?. Can not continue." ; exit 1; fi
		fi
	done

	build_install_legacy_gim
	powercycle_st_server

	echo "vbios:" 
	sshpass -p $HOST_PW ssh -o StrictHostKeyChecking=no root@$HOST_IP "$AMDVBFLASH_PATH -i"

	echo "setting kernel 5.4.38+ and vbios to $VBIOS_5438..."
	for i in {0..14}
	do
		if [[ $CONFIG_FAKE_FLASH_VBIOS -eq 1 ]] ; then
			sleep 1
			echo sshpass -p $HOST_PW ssh -o StrictHostKeyChecking=no $HOST_USER@$HOST_IP "$AMDVBFLASH_PATH -f -p $i $VBIOS_5438"
		else
			#sshpass -p $HOST_PW ssh -o StrictHostKeyChecking=no $HOST_USER@$HOST_IP "$AMDVBFLASH_PATH -f -p $i $VBIOS_415"
			#ret=$?
			#if [[ ! -z $ret ]] ; then echo "ret: $?. Can not continue." ; exit 1; fi
		fi
	done
	build_install_libgv
	powercycle_st_server

	echo "vbios:" 
	sshpass -p $HOST_PW ssh -o StrictHostKeyChecking=no root@$HOST_IP "$AMDVBFLASH_PATH -i"
done	
