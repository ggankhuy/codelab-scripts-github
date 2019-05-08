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

apt install virt-what -y

function do_inside_vm () {
	echo do_inside_vm
}

if [[ -z `which virt-what` ]] ; then
	echo "Failed to install virt-what..."
	exit 1
fi

clear

if [[ -z `virt-what` ]] ; then
        echo `hostname`
	echo "HOST: "
	echo " -----------------"
	uname -r
	echo " -----------------"
	lsb_release --all
	echo " -----------------"
	lsmod | egrep "^amdkfd|^amdgpu"
	echo " -----------------"
	modinfo amdgpu | egrep "^filename|^version"

	#  ssh to vm, vm number is specified in $1.
	
	if [[ -z $1 ]] ; then
		echo "ERROR: VM No. is not specified. Use virsh list to get index" 
		exit 1
	fi
	vmIp=`virsh domifaddr $1 | egrep "[0-9]+\.[0-9]+\." | tr -s ' ' | cut -d ' ' -f5 | cut -d '/' -f1`
	echo "VM IP: " $vmIp
fi


