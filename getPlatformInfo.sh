#	Python 3 based script to collect information.

#	HOST SYSTEMS: 
#	Host OS (if launched from HOS OS, ignored if launched from VM) 
#		BIOS version
#		CPU model
#		memory information
#		GPU information
#		OS version
#		OS kernel version
#		
#	Guest OS running 
#		BIOS version
#		CPU information (number of cores, model )
#		Memory information		
#		GPU information
#		GPU driver information (driver name, version)
#		OS version
#		OS kernel version
#		drop version (/drop/<dropname>)

source common.sh
p1=$1
SINGLE_BAR='---------------------------------------'
DOUBLE_BAR='======================================='
apt install virt-what sshpass -y 

if [[ $?  -ne 0 ]] ; then
	echo "ERROR: Failed to install packages..."
	exit 1
fi

function do_inside_vm () {
	echo do_inside_vm
}

if [[ -z `which virt-what` ]] ; then
	echo "Failed to install virt-what..."
	exit 1
fi

clear
echo $DOUBLE_BAR

if [[ -z `virt-what` ]] ; then
        echo `hostname`
	echo "HOST: "
	echo $SINGLE_BAR
	uname -r
	echo $SINGLE_BAR
	lsb_release --all
	echo $SINGLE_BAR
	lsmod | egrep "^amdkfd|^amdgpu"
	echo $SINGLE_BAR
	modinfo amdgpu | egrep "^filename|^version"
	echo $SINGLE_BAR
	dmidecode -t 0  | grep Version
	echo $SINGLE_BAR
        echo BIOS version: `dmidecode -t 0  | grep Version`

	#  ssh to vm, vm number is specified in $1.
	
	if [[ -z $1 ]] ; then
		echo $DOUBLE_BAR
		echo "ERROR: VM No. is not specified. Use virsh list to get index" 
		echo $DOUBLE_BAR
		exit 1
	fi
	vmIp=`virsh domifaddr $p1 | egrep "[0-9]+\.[0-9]+\." | tr -s ' ' | cut -d ' ' -f5 | cut -d '/' -f1`
	echo "VM IP: " $vmIp
fi

echo $DOUBLE_BAR
echo  "VM: $p1"
echo $SINGLE_BAR
sshpass -p amd1234 ssh root@$vmIp 'uname -r && lsb_release --all'
echo $SINGLE_BAR
sshpass -p amd1234 ssh root@$vmIp 'uname -r'
echo $SINGLE_BAR
sshpass -p amd1234 ssh root@$vmIp 'hostname'
echo $SINGLE_BAR
#sshpass -p amd1234 scp root@$vmIp '/work/drop*/debug-tool/amdvbflash -ai'


echo $DOUBLE_BAR




