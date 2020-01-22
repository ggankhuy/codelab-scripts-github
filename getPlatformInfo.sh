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
# apt install virt-what sshpass -y 

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
	echo "HOST IP:"		
	echo `ifconfig | grep inet | grep -v inet6 | grep -v 127.0.0.1`
	echo $SINGLE_BAR
	echo "HOST OS: 		"`lsb_release --all | grep -i description`
	echo $SINGLE_BAR
	echo "HOST KERNEL: 	"`uname -r`

	echo $SINGLE_BAR
	echo "HOST GPU: "	
	cat /sys/bus/pci/drivers/gim/gpuinfo | egrep "Name|Bus"
	cat /sys/bus/pci/drivers/gim/gpubios
        echo "HOST GPUDRIVER:   "`lsmod | egrep "^amdkfd|^amdgpu"`
        echo $SINGLE_BAR
        echo "HOST AMDGPU?:     "`modinfo amdgpu | egrep "^filename|^version"`
        echo $SINGLE_BAR
        echo "HOST GIM?:        "`modinfo gim | egrep "^filename|^version"`
        echo $SINGLE_BAR

	echo $SINGLE_BAR
	echo "HOST BIOS VER: 	"`dmidecode -t 0  | grep Version`
	echo $SINGLE_BAR

	#  ssh to vm, vm number is specified in $1.
	
	if [[ -z $1 ]] ; then
		echo $DOUBLE_BAR
		echo "ERROR: VM No. is not specified. Use virsh list to get index" 
		echo $DOUBLE_BAR
		exit 1
	fi
	vmIp=`virsh domifaddr $p1 | egrep "[0-9]+\.[0-9]+\." | tr -s ' ' | cut -d ' ' -f5 | cut -d '/' -f1`
else
	echo "ERROR: Please run from host..."
	exit 1
fi

echo $DOUBLE_BAR
echo "VM IP: 		$vmIp"

if [[ -z $vmIp ]] ; then
	echo "Error: vmIp is empty. Failed to get address, did you specify VM index correctly?"
	exit 1
fi

echo $SINGLE_BAR
echo "VM OS: 		"`sshpass -p amd1234 ssh root@$vmIp 'lsb_release --all | grep -i description'`
echo $SINGLE_BAR
echo "VM KERNEL:	"`sshpass -p amd1234 ssh root@$vmIp 'uname -r'`
echo $SINGLE_BAR
echo "VM HOSTNAME: 	"`sshpass -p amd1234 ssh root@$vmIp 'hostname'`
echo $SINGLE_BAR
echo "VM GPUDRIVER: 	"`sshpass -p amd1234 ssh root@$vmIp 'lsmod | egrep "^amdgpu"'`
echo $SINGLE_BAR
echo "VM GPUDRIVER: 	"`sshpass -p amd1234 ssh root@$vmIp 'lsmod | egrep "^amdkfd"'`
echo $SINGLE_BAR
echo "VM GPUDRIVER INFO:"`sshpass -p amd1234 ssh root@$vmIp 'modinfo amdgpu | egrep "^filename|^version"'`
echo $SINGLE_BAR


echo $DOUBLE_BAR




