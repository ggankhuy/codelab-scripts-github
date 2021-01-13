# Alternativelyboot between 5.4.38 and 4.15 kernel. 
# For 5.4.39, flash 20201023 VBIOS
# For 4.15 kernel, flash 19Q3 VBIOS.

HOST_IP="10.6.168.76"
BMC_IP="10.6.170.182"
LOOP_COUNT=3
EXTRA_SLEEP=14
VBIOS_5438=/drop/20201023/linux_host_package/vbios/V340L/D0531800.D04
VBIOS_415=/drop/drop-2019-q3-rc8-GOOD-install/drop-2019-q3-rc8/vbios/V340L/D0531800.Y03
AMDVBFLASH_PATH=/root/tools/amdvbflash/amdvbflash-4.68/amdvbflash
BMC_PW=0penBmc
BMC_USE=root
HOST_PW=amd1234
HOST_USER=root
for i in {0..3};
do
	echo "--- LOOP COUNT $i ----"

	echo "setting kernel 4.15... and vbios to $VBIOS_415"
	for i in {0..14}
	do
		#echo sshpass -p $HOST_PW ssh $HOST_USER@$HOST_IP "echo flash gpu $i ... ; $AMDVBFLASH_PATH -f -p $i $VBIOS_415"
		echo sshpass -p $HOST_PW ssh $HOST_USER@$HOST_IP "$AMDVBFLASH_PATH -f -p $i $VBIOS_415"
		sshpass -p $HOST_PW ssh $HOST_USER@$HOST_IP "$AMDVBFLASH_PATH -f -p $i $VBIOS_415"
		ret=$?
		if [[ ! -z $ret ]] ; then echo "ret: $?. Can not continue." ; exit 1; fi
	done
	sshpass -p $HOST_PW ssh $HOST_USER@$HOST_IP 'cp /root/grub.4.15.log /etc/default/grub ; update-grub'
	sleep $EXTRA_SLEEP
	echo "sshpass -p $BMC_PW ssh $BMC_USER@$BMC_IP 'cd ~/BMC_Scripts/ ; python shut_down_system.py ; sleep 15; python boot_up_system.py'"
	sshpass -p $BMC_PW ssh $BMC_USER@$BMC_IP 'cd ~/BMC_Scripts/ ; python shut_down_system.py ; sleep 15; python boot_up_system.py'
	if [[ $? -ne 0 ]] ; then echo "BMC is not pingable, can not continue..." ; exit 1; fi
	sleep $EXTRA_SLEEP
	for i in {0..60} ; 
	do
		ping -c 1 $HOST_IP 
		if [[ $? -eq 0 ]] ; then
			echo "system is pingable, give another $EXTRA_SLEEP sec..."
			sleep $EXTRA_SLEEP
			break
		fi
		sleep 5
	done	
	echo "kernel: "
	sshpass -p amd1234 ssh root@$HOST_IP 'uname -r'
	echo "vbios:" 
	sshpass -p $HOST_PW ssh root@$HOST_IP "$AMDVBFLASH_PATH -i"

	echo "setting kernel 5.4.38+ and vbios to $VBIOS_5438..."
	for i in {0..14}
	do
		sshpass -p $HOST_PW ssh root@$HOST_IP "echo flash gpu $i ... ; $AMDVBFLASH_PATH -f -p $i $VBIOS_5438"
	done
	sshpass -p $HOST_PW ssh root@$HOST_IP 'cp grub.5.4.38.log /etc/default/grub ; update-grub'
	sleep $EXTRA_SLEEP
	sshpass -p $BMC_PW ssh $BMC_USER@$BMC_IP 'cd ~/BMC_Scripts/ ; python shut_down_system.py ; sleep 15; python boot_up_system.py'
	if [[ $? -ne 0 ]] ; then echo "BMC is not pingable, can not continue..." ; exit 1; fi
	sleep $EXTRA_SLEEP
	for i in {0..60} ; 
	do
		ping -c 1 $HOST_IP 
		if [[ $? -eq 0 ]] ; then
			echo "system is pingable, give another 15 sec..."
			sleep $EXTRA_SLEEP
			break
		fi
		sleep 5
	done	
	echo "kernel: "
	sshpass -p $HOST_PW ssh root@$HOST_IP 'uname -r'
	echo "vbios:" 
	sshpass -p $HOST_PW ssh root@$HOST_IP "$AMDVBFLASH_PATH -i"
done	
