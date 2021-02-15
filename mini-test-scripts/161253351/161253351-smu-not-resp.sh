# Alternatively boot between legacy and libgv gim version between powercycle.
# Synopsis:
# Prerequisites for running the script:
# - Verify you installed the vats2 (20201023)
# - - Verify smi-lib available. 
# - Make available amdvbflash tool and set path in AMDVBFLASH_PATH
# - Make available the vbios binary path and set path in VBIOS_5438 and VBIOS_415 (optional)

# 1. For every other boot build and install libgv and legacy-gim on other boots.
# 2. For libgv flash latest libgv compatible VBIOS (i.e. currently D04), for legacy-gim, flash load Y03 VBIOS.
# 3. Powercycle (for G servers) or reboot (for non-G servers)
# 3. Load gim  (libgv or legacy-gim)
# 4. Start all VM-s

DATE=`date +%Y%m%d-%H-%M-%S`

#	IXT70 configuration

HOST_IP="10.216.66.51"
VBIOS_5438=/drop/20201023/linux_host_package/vbios/V340L/D0531800.D04
VBIOS_415=/drop/drop-2019-q3-rc8-GOOD-install/drop-2019-q3-rc8/vbios/V340L/D0531800.Y03

#	ST SEMI02 configuration

HOST_IP="10.216.52.50"
BMC_IP="10.216.52.51"
VBIOS_5438=/drop/20201023/linux_host_package/vbios/NAVI12/Gemini
VBIOS_415=$VBIOS_5438

# 	SJC SEMI05 configuration

HOST_IP="10.216.52.74"
BMC_IP="10.216.52.75"
VBIOS_5438=/drop/20201023/linux_host_package/vbios/V340L/D0531800.D04
VBIOS_415=/drop/drop-2019-q3-rc8-GOOD-install/drop-2019-q3-rc8/vbios/V340L/D0531800.Y03

# 	MKM ST 47 configuration

HOST_IP="10.6.168.76"
BMC_IP="10.6.170.182"
#VBIOS_5438=/drop/20201023/linux_host_package/vbios/V340L/D0531800.D04
#VBIOS_415=/drop/drop-2019-q3-rc8-GOOD-install/drop-2019-q3-rc8/vbios/V340L/D0531800.Y03
VBIOS_5438=""
VBIOS_415=""

#	IXT39 configuration

HOST_IP="10.216.66.54"
VBIOS_5438=/drop/20201023/linux_host_package/vbios/V340L/D0531800.D04
VBIOS_415=/drop/drop-2019-q3-rc8-GOOD-install/drop-2019-q3-rc8/vbios/V340L/D0531800.Y03

#	misc. configuration 

LOOP_COUNT=0
EXTRA_SLEEP=120
AMDVBFLASH_PATH=/root/tools/amdvbflash/amdvbflash-4.68/amdvbflash

BMC_PW=0penBmc
BMC_USER=root
HOST_PW=amd1234
HOST_USER=root

HOST_RESPONSIVE=0
CONFIG_DISABLE_FLASH_VBIOS=1
CONFIG_DISABLE_HOST_DRV_BUILD=0
CONFIG_LAUNCH_MONITOR=1
CONFIG_LAUNCH_VK_EXAMPLE=1
DROP_FOLDER_ROOT=/drop/20201023/
VK_EXAMPLE_SCRIPT=vk-example-launch.sh
RUN_TEST_SCRIPT=run-test-161253351.sh

DIRNAME=161253351-result/$DATE-$HOST_IP
mkdir -p $DIRNAME

echo "host information: $HOST_IP, $HOST_USER, $HOST_PW, VBIOS_415: $VBIOS_415, VBIOS_5438: $VBIOS_5438"
sleep 1

#	Reboot the server instead of powercycle. Powercycle is only supported on ST or other G servers.
#	If you set the CONFIG_REBOOT=0 and if it is not G server, result is not predictable.
#	Powercycle is preferable test option over reboot. 

CONFIG_REBOOT=1

#   Unload gim.
#   Clear dmesg.
#   Load gim.
#   Output dmesg after gim load to $DIRNAME/summary.log
#   Output dmesg after gim load to $DIRNAME/dmesg-modprobe-libgv.$loopCnt.log
#   Clear dmesg.
#   Start all VM-s
#   Output dmesg after all vm start to $DIRNAME/dmesg-start-vm.libgv.$loopCnt.log

function output_stat_libgv {
   	echo "vbios/gim:" 
	sshpass -p $HOST_PW ssh -o StrictHostKeyChecking=no $HOST_USER@$HOST_IP "$AMDVBFLASH_PATH -i" | tee -a $DIRNAME/summary.log
    echo " --- loading libgv and save the dmesg ---" | tee -a $DIRNAME/$loopCnt.log
    echo " --- loading libgv and save the dmesg ---" > $DIRNAME/dmesg-libgv-build-load.$loopCnt.log
	sshpass -p $HOST_PW ssh -o StrictHostKeyChecking=no $HOST_USER@$HOST_IP "modprobe -r gim ; dmesg --clear ; modprobe gim ; dmesg | grep \"GPU IOV MODULE\"" > $DIRNAME/summary.log
	sshpass -p $HOST_PW ssh -o StrictHostKeyChecking=no $HOST_USER@$HOST_IP "modprobe -r gim ; dmesg --clear ; modprobe gim ; dmesg | grep \"GPU IOV MODULE\"" >> $DIRNAME/dmesg-gim-build-load.$loopCnt.log
	echo " --- dmesg after modprobe libgv ---" | tee -a $DIRNAME/$loopCnt.log
	echo " --- dmesg after modprobe libgv ---" > $DIRNAME/dmesg-modprobe-libgv.$loopCnt.log
	sshpass -p $HOST_PW ssh -o StrictHostKeyChecking=no $HOST_USER@$HOST_IP "dmesg" >>  $DIRNAME/$loopCnt.log
	sshpass -p $HOST_PW ssh -o StrictHostKeyChecking=no $HOST_USER@$HOST_IP "dmesg" >>  $DIRNAME/dmesg-modprobe-libgv.$loopCnt.log
	echo " --- dmesg after start all VM-S (libgv loaded) ---" | tee -a $DIRNAME/$loopCnt.log
	echo " --- dmesg after start all VM-S (libgv loaded) ---" > $DIRNAME/dmesg-start-vm.libgv.$loopCnt.log
	#sshpass -p $HOST_PW ssh -o StrictHostKeyChecking=no $HOST_USER@$HOST_IP 'echo virsh net-start default; dmesg --clear ; for k in $(seq 0 $gpu_count) ; do virsh start vats-test-0$k ; done'
	sshpass -p $HOST_PW ssh -o StrictHostKeyChecking=no $HOST_USER@$HOST_IP 'echo virsh net-start default; dmesg --clear'
	#sshpass -p $HOST_PW ssh -o StrictHostKeyChecking=no $HOST_USER@$HOST_IP 'virsh net-start default; dmesg --clear ; for k in $(seq 0 $gpu_count) ; do virsh start vats-test-0$k ; done'
	#sshpass -p $HOST_PW ssh -o StrictHostKeyChecking=no $HOST_USER@$HOST_IP 'dmesg' >>  $DIRNAME/$loopCnt.log
	#sshpass -p $HOST_PW ssh -o StrictHostKeyChecking=no $HOST_USER@$HOST_IP 'dmesg' >>  $DIRNAME/dmesg-start-vm.libgv.$loopCnt.log
}
function output_stat {
   	echo "vbios/gim:" 
	sshpass -p $HOST_PW ssh -o StrictHostKeyChecking=no $HOST_USER@$HOST_IP "$AMDVBFLASH_PATH -i" | tee -a $DIRNAME/summary.log
    echo " --- Building gim, loading and save the dmesg ---" | tee -a $DIRNAME/$loopCnt.log
    echo " --- Building gim, loading and save the dmesg ---" > $DIRNAME/dmesg-gim-build-load.$loopCnt.log
	sshpass -p $HOST_PW ssh -o StrictHostKeyChecking=no $HOST_USER@$HOST_IP "modprobe -r gim ; dmesg --clear ; modprobe gim ; dmesg | grep \"GPU IOV MODULE\"" > $DIRNAME/summary.log
	sshpass -p $HOST_PW ssh -o StrictHostKeyChecking=no $HOST_USER@$HOST_IP "modprobe -r gim ; dmesg --clear ; modprobe gim ; dmesg | grep \"GPU IOV MODULE\"" >> $DIRNAME/dmesg-gim-build-load.$loopCnt.log
	echo " --- dmesg after modprobe gim-legacy ---" | tee -a $DIRNAME/$loopCnt.log
	echo " --- dmesg after modprobe gim-legacy ---" > $DIRNAME/dmesg-modprobe-gim.$loopCnt.log
	sshpass -p $HOST_PW ssh -o StrictHostKeyChecking=no $HOST_USER@$HOST_IP "dmesg" >>  $DIRNAME/$loopCnt.log
	sshpass -p $HOST_PW ssh -o StrictHostKeyChecking=no $HOST_USER@$HOST_IP "dmesg" >>  $DIRNAME/dmesg-modprobe-gim.$loopCnt.log
	echo " --- dmesg after start all VM-S ---" | tee -a $DIRNAME/$loopCnt.log
	echo " --- dmesg after start all VM-S ---" > $DIRNAME/dmesg-start-vm.$loopCnt.log
	sshpass -p $HOST_PW ssh -o StrictHostKeyChecking=no $HOST_USER@$HOST_IP 'virsh net-start default; dmesg --clear ; for k in $(seq 0 $gpu_count) ; do virsh start vats-test-0$k ; done ; dmesg' >>  $DIRNAME/$loopCnt.log
	sshpass -p $HOST_PW ssh -o StrictHostKeyChecking=no $HOST_USER@$HOST_IP 'virsh net-start default; dmesg --clear ; for k in $(seq 0 $gpu_count) ; do virsh start vats-test-0$k ; done ; dmesg' >>  $DIRNAME/dmesg-start-vm.$loopCnt.log
}

function build_install_legacy_gim() {
	if [[ $CONFIG_DISABLE_HOST_DRV_BUILD -ne 1 ]] ; then
		cmds=( "mkdir /git.co/" "cd /git.co/ ; git clone https://ggghamd:amd1234A%23@github.com/ggghamd/cp-Gibraltar-GIM.git"  \
			"cd /git.co/cp-Gibraltar-GIM; ./dkms-gim.sh gim 1.0" "modprobe gim")
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

#   counter adapters

ret=`sshpass -p $HOST_PW ssh -o StrictHostKeyChecking=no $HOST_USER@$HOST_IP '/root/tools/amdvbflash/amdvbflash-4.68/amdvbflash -i | grep AMDVBFLASH | wc -l'`
echo $ret

if [[ $ret == 0 ]] ; then
    echo "$AMDVBFLASH_PATH can not be found on $HOST_IP..."
    exit 1
else    
    echo "$AMDVBFLASH_PATH found, OK..."
fi

gpu_count=`sshpass -p $HOST_PW ssh -o StrictHostKeyChecking=no $HOST_USER@$HOST_IP "$AMDVBFLASH_PATH -i |  grep adapter -A 20 | wc -l"`
gpu_count=$((gpu_count-2))
echo "No. of adapters: $gpu_count"

echo "Copying artifacts to target system:"
for i in monitor.sh $VK_EXAMPLE_SCRIPT $RUN_TEST_SCRIPT
do
    echo "copying $i..."
    sshpass -p amd1234 scp -o StrictHostKeyChecking=no $i $HOST_USER@$HOST_IP:/root/
done

for loopCnt in $(seq 0 $LOOP_COUNT) ;
do
	echo "--- LOOP COUNT $loopCnt ----" | tee -a $DIRNAME/summary.log

    output_stat_libgv
    #echo "Sleeping for 30 minutes..."
    #sleep 1800

    if [[ $CONFIG_LAUNCH_MONITOR -eq 1 ]] ; then
        echo "Launching monitor.sh on target..."
        CONFIG_FOLDER_MONITOR_APP=/usr/src/gim-2.0.1.G.20201023/smi-lib/examples/monitor/
        ret=`sshpass -p $HOST_PW ssh -o StrictHostKeyChecking=no $HOST_USER@$HOST_IP "ps -ax | grep monitor.sh | grep -v grep"`
        if [[ -z $ret ]] ; then
            echo "monitor.sh is not running on the host, launching..."
            sshpass -p $HOST_PW ssh -o StrictHostKeyChecking=no $HOST_USER@$HOST_IP "cp /root/monitor.sh $CONFIG_FOLDER_MONITOR_APP ; cd $CONFIG_FOLDER_MONITOR_APP ; make ; nohup ./monitor.sh > monitor.log &"
        else
            echo "monitor.sh is already running on the host. Bypassing the monitor.sh launch: $ret"
        fi
    else
        echo "Bypassing monitor.sh launch target..."
    fi

    # launch vk examples for N hours.

    if [[ $CONFIG_LAUNCH_VK_EXAMPLE -eq 1 ]] ; then

        echo "launching vk examples..."
    	
        #sshpass -p $HOST_PW ssh -o StrictHostKeyChecking=no $HOST_USER@$HOST_IP \
        #"cd $DROP_FOLDER_ROOT ; pwd; nohup ./$VK_EXAMPLE_SCRIPT > ./output.$loopCnt.log &"
        
    	sshpass -p $HOST_PW ssh -o StrictHostKeyChecking=no $HOST_USER@$HOST_IP \
        "mv /root/$RUN_TEST_SCRIPT /drop/20201023 ; cd /drop/20201023 ; nohup ./$RUN_TEST_SCRIPT &"
        
        # For ever 15 minutes. Check dmesg log.

        # powercycle_server
    fi

done	
echo "done..."
