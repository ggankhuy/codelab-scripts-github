#	Python 3 based script to cosystem llect information.
# Compatibility:
# Ubuntu 16.x, 19.x 
# Centos, Redhat (future enhancement).
# Requirement:
# For VM information, virsh utility is used.
# For VEGA10 cards, the sysfs files queried.
# For NAVIx cards, the libsmi calls are queried. (future enhancement).

#	Uti/clitiesommands used:
# virsh (vm info)
# virt-what (check virtual env)
# dmidecode (smbios)
# ifconfig (network)
# modinfo
# hostname
# uname
# lsb_release 
# lsmod

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

#source common.sh
p1=$1
SINGLE_BAR='---------------------------------------'
DOUBLE_BAR='======================================='
DATE=`date +%Y%m%d-%H-%M-%S`
CONFIG_PATH_PLAT_INFO=/plat-info/$DATE/
CONFIG_FILE_PLAT_INFO=$CONFIG_PATH_PLAT_INFO/$DATE-platform-info.log
CONFIG_FILE_DMESG_HOST=$CONFIG_PATH_PLAT_INFO/$DATE-dmesg-host.log
CONFIG_FILE_DMESG_GUEST=$CONFIG_PATH_PLAT_INFO/$DATE-dmesg-guest-$p1.log
CONFIG_FILE_CLINFO_GUEST=$CONFIG_PATH_PLAT_INFO/$DATE-dmesg-clinfo-$p1.log
CONFIG_FILE_MODINFO_AMDGPU_GUEST=$CONFIG_PATH_PLAT_INFO/$DATE-modinfo-amdgpu-$p1.log

function host_guest_1()  {
	echo $SINGLE_BAR | tee $CONFIG_FILE_PLAT_INFO
	echo "HOST BIOS VER: 	"`dmidecode -t 0  | grep Version` | tee $CONFIG_FILE_PLAT_INFO
	echo $SINGLE_BAR | tee $CONFIG_FILE_PLAT_INFO
	echo "HOST GCC VER: " `gcc --version | grep gcc` | tee $CONFIG_FILE_PLAT_INFO
	echo "HOST G++ VER: " `g++ --version | grep g++` | tee $CONFIG_FILE_PLAT_INFO
	echo "HOST C++ VER: " `c++ --version | grep c++` | tee $CONFIG_FILE_PLAT_INFO
	echo "HOST CC VER: " `cc --version | grep cc` | tee $CONFIG_FILE_PLAT_INFO
	echo "CLANG VER: " `clang --version | grep clang` | tee $CONFIG_FILE_PLAT_INFO
	echo "VIRT. SW VER: " `virsh --version` | tee $CONFIG_FILE_PLAT_INFO
	
}


if [[ $p1 == "--help" ]] ; then
	clear
	echo "usage: "
	echo "$0 - get host information without specifying VM"
	echo "$0 <vm_index> get host and guest information. Use virsh to get vm index."
	exit 0
fi
mkdir -p $CONFIG_PATH_PLAT_INFO

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
echo $DOUBLE_BAR | tee $CONFIG_FILE_PLAT_INFO

# Set distribution specific commands here.
DISTRO=`cat /etc/os-release | grep ^ID=  |  cut -d '=' -f2`
if [[ -z $DISTRO ]] ; then
	echo "Can not determine which distribution is."
	exit 0
fi

if [[ $DISTRO == "ubuntu" ]] ; then
	echo "Ubuntu distribution detected."
elif [[ $DISTRO == "cent" ]] ; then
	echo "CENTOS/REDHAT distribution detected."
else
	echo "Unknown distribution"
	exit 0
fi

# -----------------------------------------
# Get host information
# -----------------------------------------

if [[ -z `virt-what` ]] ; then
        echo `hostname`
	echo "HOST: " | tee $CONFIG_FILE_PLAT_INFO
	echo $SINGLE_BAR | tee $CONFIG_FILE_PLAT_INFO
	echo "HOST IP:" | tee $CONFIG_FILE_PLAT_INFO
	echo `ifconfig | grep inet | grep -v inet6 | grep -v 127.0.0.1` | tee $CONFIG_FILE_PLAT_INFO
	echo $SINGLE_BAR | tee $CONFIG_FILE_PLAT_INFO
	echo "HOST OS:          "`lsb_release --all | grep -i description` | tee $CONFIG_FILE_PLAT_INFO
	echo $SINGLE_BAR | tee $CONFIG_FILE_PLAT_INFO
	echo "HOST KERNEL: 	"`uname -r` | tee $CONFIG_FILE_PLAT_INFO

	echo $SINGLE_BAR | tee $CONFIG_FILE_PLAT_INFO
	echo "HOST GPU: " | tee $CONFIG_FILE_PLAT_INFO
	cat /sys/bus/pci/drivers/gim/gpuinfo | egrep "Name|Bus" | tee $CONFIG_FILE_PLAT_INFO
	cat /sys/bus/pci/drivers/gim/gpubios | tee $CONFIG_FILE_PLAT_INFO
        echo "HOST GPUDRIVER:   "`lsmod | egrep "^amdkfd|^amdgpu"` | tee $CONFIG_FILE_PLAT_INFO
        echo $SINGLE_BAR | tee $CONFIG_FILE_PLAT_INFO
        echo "HOST AMDGPU?:     "`modinfo amdgpu | egrep "^filename|^version"` | tee $CONFIG_FILE_PLAT_INFO
        echo $SINGLE_BAR | tee $CONFIG_FILE_PLAT_INFO
        echo "HOST GIM?:        "`modinfo gim | egrep "^filename|^version"` | tee $CONFIG_FILE_PLAT_INFO

	host_guest_1

	#  ssh to vm, vm number is specified in $1.
	
	if [[ -z $1 ]] ; then
		echo $DOUBLE_BAR | tee $CONFIG_FILE_PLAT_INFO
		echo "VM No. is not specified. "
		echo $DOUBLE_BAR | tee $CONFIG_FILE_PLAT_INFO
		vmpIp=""
	else
		vmIp=`virsh domifaddr $p1 | egrep "[0-9]+\.[0-9]+\." | tr -s ' ' | cut -d ' ' -f5 | cut -d '/' -f1`
		if [[ -z $vmIp ]] ; then
			echo "Use virsh to determine running VM-s indices."
		exit 1
		fi
	fi
	dmesg >> $CONFIG_FILE_DMESG_HOST
else
	echo "ERROR: Please run from host..." 
	exit 1
fi

# -----------------------------------------
# Get guest information
# -----------------------------------------

echo $DOUBLE_BAR | tee $CONFIG_FILE_PLAT_INFO
echo "VM IP:" 		$vmIp| tee $CONFIG_FILE_PLAT_INFO

if [[ -z $vmIp ]] ; then
	echo "vmIp is empty. Either failed to get address or did not specify vm index."
else

	echo $SINGLE_BAR | tee $CONFIG_FILE_PLAT_INFO
	echo "VM OS: 		"`sshpass -p amd1234 ssh root@$vmIp 'lsb_release --all | grep -i description'`| tee $CONFIG_FILE_PLAT_INFO
	echo $SINGLE_BAR | tee $CONFIG_FILE_PLAT_INFO
	echo "VM KERNEL:	"`sshpass -p amd1234 ssh root@$vmIp 'uname -r'` | tee $CONFIG_FILE_PLAT_INFO
	echo $SINGLE_BAR | tee $CONFIG_FILE_PLAT_INFO
	echo "VM HOSTNAME: 	"`sshpass -p amd1234 ssh root@$vmIp 'hostname'` | tee $CONFIG_FILE_PLAT_INFO
	echo $SINGLE_BAR | tee $CONFIG_FILE_PLAT_INFO
	echo "VM GPUDRIVER INFO:"`sshpass -p amd1234 ssh root@$vmIp 'modinfo amdgpu | egrep "^filename|^version"'` | tee $CONFIG_FILE_PLAT_INFO
	echo $SINGLE_BAR | tee $CONFIG_FILE_PLAT_INFO
	sshpass -p amd1234 ssh root@$vmIp 'dmesg' >  $CONFIG_FILE_DMESG_GUEST
	sshpass -p amd1234 ssh root@$vmIp 'modinfo amdgpu' >  $CONFIG_FILE_MODINFO_AMDGPU_GUEST
	echo $SINGLE_BAR | tee $CONFIG_FILE_PLAT_INFO
	sshpass -p amd1234 ssh root@$vmIp 'clinfo' >  $CONFIG_FILE_CLINFO_GUEST
	echo "CLINFO VERSION:" `sshpass -p amd1234 ssh root@$vmIp 'clinfo -v'` | tee  $CONFIG_FILE_PLAT_INFO
	echo $SINGLE_BAR | tee $CONFIG_FILE_PLAT_INFO
	host_guest_1
fi

echo $DOUBLE_BAR | tee $CONFIG_FILE_PLAT_INFO

echo LOG FILES ARE STORED AT: $CONFIG_PATH_PLAT_INFO:
ls -l $CONFIG_PATH_PLAT_INFO



