# Platform info script to gather system information.
#!/usr/bin/sh


# Usage:
# techsupport.sh - to display host information.
# techsupport.sh <VMno> - to display both host and guest information.

# Requirements/assumptions:
#   - To run this script, either amdgpu or libgv must be installed and loaded and this script will determine 
#   the presence using dkms status and gather respective logs.
#   - if in instance, both amdgpu and libgv(gim) installed, it will gather only libgv information and ignore amdgpu.
#   - To gather guest log, guest VM must be running and amdgpu on guest must be loaded.
#   - libgv and gim is used interchangeably, meaning kvm based virtualization drivers.

# log file structure:
# ts/
#       <DATE>
#           <DATE>-platform-summary log (containing overall information on host + guest(if applicable)
# <DATE->-techsupport.tar
#           host part: cpu, gpu model, amdgpu/libgv version, kernel versO/Sn, host os version.
#           guest part: cpu, gpu mod(guest) version, kernel version, guest O/S version.
#       host/
#           host logs: dmesg, modi(or libgv) info amdgpu, amdgpu(or libgv) parameters, rocminfo, rocm-smi.
#       guest/
#           guest logs: dmesg, modinfo amdgpu, amdgpu parameters, rocminfo, rocm-smi.


p1=$1
DEBUG=1
SINGLE_BAR='---------------------------------------'
DOUBLE_BAR='======================================='
DATE=`date +%Y%m%d-%H-%M-%S`
CONFIG_PATH_PLAT_INFO=./ts/$DATE
CONFIG_FILE_TAR=./ts/$DATE/$DATE-techsupport.tar
CONFIG_FILE_PLAT_INFO=$CONFIG_PATH_PLAT_INFO/$CONFIG_SUBDIR_HOST/$DATE-platform-info.log
CONFIG_SUBDIR_HOST=host
CONFIG_SUBDIR_GUEST=guest

CONFIG_FILE_PARM_AMDGPU_HOST=$CONFIG_PATH_PLAT_INFO/$CONFIG_SUBDIR_HOST/$DATE-parm-amdgpu-host.log
CONFIG_FILE_PARM_LIBGV_HOST=$CONFIG_PATH_PLAT_INFO/$CONFIG_SUBDIR_HOST/$DATE-parm-libgv-host.log

CONFIG_FILE_MODINFO_AMDGPU_HOST=$CONFIG_PATH_PLAT_INFO/$CONFIG_SUBDIR_HOST/$DATE-modinfo-amdgpu-host.log
CONFIG_FILE_MODINFO_LIBGV_HOST=$CONFIG_PATH_PLAT_INFO/$CONFIG_SUBDIR_HOST/$DATE-modinfo-libgv-host.log

CONFIG_FILE_DMESG_HOST=$CONFIG_PATH_PLAT_INFO/$CONFIG_SUBDIR_HOST/$DATE-dmesg-host.log
CONFIG_FILE_SYSLOG_HOST=$CONFIG_PATH_PLAT_INFO/$CONFIG_SUBDIR_HOST/$DATE-syslog-host.log
CONFIG_FILE_KERN_LOG_HOST=$CONFIG_PATH_PLAT_INFO/$CONFIG_SUBDIR_HOST/$DATE-kernlog-host.log
CONFIG_FILE_DMIDECODE_HOST=$CONFIG_PATH_PLAT_INFO/$CONFIG_SUBDIR_HOST/$DATE-dmidecode-host.log
CONIG_FILE_ROCMINFO_HOST=$CONFIG_PATH_PLAT_INFO/$CONFIG_SUBDIR_HOST/$DATE-rocminfo-host.log
CONFIG_FILE_ROCMSMI_HOST=$CONFIG_PATH_PLAT_INFO/$CONFIG_SUBDIR_HOST/$DATE-rocm-smi-host.log

CONFIG_FILE_DMESG_GUEST=$CONFIG_PATH_PLAT_INFO/$CONFIG_SUBDIR_GUEST/$DATE-dmesg-guest-$p1.log
CONFIG_FILE_CLINFO_GUEST=$CONFIG_PATH_PLAT_INFO/$CONFIG_SUBDIR_GUEST/$DATE-dmesg-clinfo-$p1.log
CONFIG_FILE_MODINFO_AMDGPU_GUEST=$CONFIG_PATH_PLAT_INFO/$CONFIG_SUBDIR_GUEST/$DATE-modinfo-amdgpu-$p1.log
CONFIG_FILE_SYSLOG_GUEST=$CONFIG_PATH_PLAT_INFO/$CONFIG_SUBDIR_GUEST/$DATE-syslog-guest.log
CONFIG_FILE_KERN_LOG_GUEST=$CONFIG_PATH_PLAT_INFO/$CONFIG_SUBDIR_HOST/$DATE-kernlog-guest.log

mkdir -p $CONFIG_PATH_PLAT_INFO/$CONFIG_SUBDIR_GUEST
mkdir -p $CONFIG_PATH_PLAT_INFO/$CONFIG_SUBDIR_HOST

function host_guest_1()  {
	echo $SINGLE_BAR | tee -a $CONFIG_FILE_PLAT_INFO
	echo $SINGLE_BAR | tee -a $CONFIG_FILE_PLAT_INFO
	echo "HOST BIOS VER: 	"`dmidecode -t 0  | grep Version` | tee -a $CONFIG_FILE_PLAT_INFO
	echo $SINGLE_BAR | tee -a $CONFIG_FILE_PLAT_INFO
	echo "HOST GCC VER: " `gcc --version | grep gcc` | tee -a $CONFIG_FILE_PLAT_INFO
	echo "HOST G++ VER: " `g++ --version | grep g++` | tee -a $CONFIG_FILE_PLAT_INFO
	echo "HOST C++ VER: " `c++ --version | grep c++` | tee -a $CONFIG_FILE_PLAT_INFO
	echo "HOST CC VER: " `cc --version | grep cc` | tee -a $CONFIG_FILE_PLAT_INFO
	echo "HOST HCC VER: " `hcc --version | grep hcc` | tee -a $CONFIG_FILE_PLAT_INFO
	echo "CLANG VER: " `clang --version | grep clang` | tee -a $CONFIG_FILE_PLAT_INFO
	echo "VIRT. SW VER: " `virsh --version` | tee -a $CONFIG_FILE_PLAT_INFO
	
}

#   Start gathering logs.
#   If p1 is not specified, gather only host system.
#       Determine if gim (libgv) installed (virtualized) or amdgpu installed (baremetal).
#       amdgpu compatible logs (baremetal):
#       libgv compatible logs (virt).
#   else if p1 is specified and libgv is not installed, exit with error or optionally, display host with amdgpu info only.
#       libgv compatible logs (virt) from host.
#       gather guest logs.
#

function ts_amdgpu_compat_full_logs() {
    ts_helper_full_logs amdgpu
}

function ts_libgv_compat_full_logs() {
    ts_helper_full_logs libgv
}

function ts_helper_full_logs() {
    p1=$1
    if [[ $DEBUG ]] ; then
        echo "ts_helper_summary_logs entered..."
        echo "p1: $p1"
    fi

    dmesg | tee $CONFIG_FILE_DMESG_HOST
    cat /var/log/syslog | tee $CONFIG_FILE_SYSLOG_HOST
	cat /var/log/kern.log | tee $CONFIG_FILE_KERN_LOG_HOST
    dmidecode | tee $CONFIG_FILE_DMIDECODE_HOST

    if [[ $p1 == "amdgpu" ]] ; then
        modinfo amdgpu | tee $CONFIG_FILE_MODINFO_AMDGPU_HOST
        for i in /sys/module/amdgpu/parameters/* ; do echo -n $i: ; cat $i ; done 2>&1 | tee $CONFIG_FILE_PARM_AMDGPU_HOST 
        rocminfo 2>&1 | tee $CONFIG_FILE_ROCMINFO_HOST
        rocm-smi --showall 2>&1 | tee $CONFIG_FILE_ROCMSMI_HOST
    elif [[ $p1 == "libgv" ]] ; then
        modinfo gim | tee $CONFIG_FILE_MODINFO_LIBGV_HOST
        for i in /sys/module/gim/parameters/* ; do echo -n $i: ; cat $i ; done 2>&1 | tee $CONFIG_FILE_PARM_LIBGV_HOST 
    elif [[ $p1 = "" ]] ; then
        echo "Warning: p1 is empty, neither amdgpu or libgv specific logs will be gathered."
    else
        echo "warning: Unknown parameter! "
    fi
}
function ts_helper_summary_logs() {
    p1=$1
    if [[ $DEBUG ]] ; then
        echo "ts_helper_summary_logs entered..."
        echo "p1: $p1"
    fi

	echo    "HOSTNAME:  " `hostname` | tee $CONFIG_FILE_PLAT_INFO
	echo $SINGLE_BAR | tee -a $CONFIG_FILE_PLAT_INFO

 	echo -n "IP:        " | tee -a $CONFIG_FILE_PLAT_INFO
    if [[  `which ifconfig` ]] ; then
	    echo `ifconfig | grep inet | grep -v inet6 | grep -v 127.0.0.1` | tee -a $CONFIG_FILE_PLAT_INFO
    else
        echo "ifconfig not available" | tee -a $CONFIG_FILE_PLAT_INFO
    fi

	echo $SINGLE_BAR | tee -a $CONFIG_FILE_PLAT_INFO
	echo "O/S:          "`cat /etc/os-release  | grep PRETTY_NAME` | tee -a $CONFIG_FILE_PLAT_INFO

	echo $SINGLE_BAR | tee -a $CONFIG_FILE_PLAT_INFO
	echo "O/S KERNEL: 	"`uname -r` | tee -a $CONFIG_FILE_PLAT_INFO

    if [[ $p1 == "amdgpu" ]] ; then
    	echo $SINGLE_BAR | tee -a $CONFIG_FILE_PLAT_INFO
	    echo "GPU:      " | tee -a $CONFIG_FILE_PLAT_INFO
        if [[ `which rocm-smi` ]] ; then
            rocm-smi --showall | grep "series:" 2>&1 | tee -a $CONFIG_FILE_PLAT_INFO
            elif [[ `rocminfo` ]] ; then
            rocminfo | grep gfx 2>&1 | tee -a $CONFIG_FILE_PLAT_INFO
        elif [[ `lspci` ]] ; then
            lspci | grep Disp 2>&1 | tee -a $CONFIG_FILE_PLAT_INFO
        else
            echo "Unable to determine GPU, neither rocm-smi, rocminfo or lspci utilities available" | tee -a $CONFIG_FILE_PLAT_INFO
        fi

        echo $SINGLE_BAR | tee -a $CONFIG_FILE_PLAT_INFO
        echo "GPUDRIVER:   "`lsmod | egrep "^amdkfd|^amdgpu"` | tee -a $CONFIG_FILE_PLAT_INFO
        modinfo amdgpu | egrep "^filename|^version" | tee -a $CONFIG_FILE_PLAT_INFO
        echo $SINGLE_BAR | tee -a $CONFIG_FILE_PLAT_INFO
        echo "GPUDRIVER(DKMS):   "`dkms status | grep amdgpu` | tee -a $CONFIG_FILE_PLAT_INFO

        echo $SINGLE_BAR | tee -a $CONFIG_FILE_PLAT_INFO

        echo -n "ROCM VERSION: " | tee -a $CONFIG_FILE_PLAT_INFO
        if [[ -f /opt/rocm/.info/version ]] ; then
            cat /opt/rocm/.info/version | tee -a $CONFIG_FILE_PLAT_INFO
        else
            echo "Unable to determine rocm version. Is ROCm installed?"
        fi
        echo $SINGLE_BAR | tee -a $CONFIG_FILE_PLAT_INFO
    elif [[ $p1 == "libgv" ]] ; then
        echo $SINGLE_BAR | tee -a $CONFIG_FILE_PLAT_INFO
        echo "GPUDRIVER:   "`lsmod | egrep "^gim|^gim"` | tee -a $CONFIG_FILE_PLAT_INFO
        modinfo gim | egrep "^filename|^version" | tee -a $CONFIG_FILE_PLAT_INFO
        echo $SINGLE_BAR | tee -a $CONFIG_FILE_PLAT_INFO
        echo "GPUDRIVER(DKMS):   "`dkms status | grep gim` | tee -a $CONFIG_FILE_PLAT_INFO
        echo $SINGLE_BAR | tee -a $CONFIG_FILE_PLAT_INFO
    elif [[ $p1 = "" ]] ; then
        echo "Warning: p1 is empty, neither amdgpu or libgv specific logs will be gathered."
    else
        echo "warning: Unknown parameter! "
    fi
}
function ts_amdgpu_compat_summary_logs() {
    ts_helper_summary_logs amdgpu
}
function ts_libgv_compat_summary_logs() {
    ts_helper_summary_logs libgv
}

function ts_guest() {
    echo "ts_guest..."
}

AMDGPU_PRESENCE=`dkms status | grep amdgpu`
LIBGV_PRESENCE=`dkms status | grep gim`
echo "AMDGPU_PRESENCE: $AMDGPU_PRESENCE"
echo "LIBGV_PRESENCE: $LIBGV_PRESENCE"

# If both libgv and amdgpu present, libgv will take presedence...

if [[ $LIBGV_PRESENCE ]] ; then
    echo "Gathering libgv compatible logs..."
    ts_libgv_compat_summary_logs
    ts_libgv_compat_full_logs
    if [[ $1 ]] ; then
        echo "Gathering guest log: "
        ts_guest
        ts_libgv_compat_summary_logs
    fi
elif [[ $AMDGPU_PRESENCE ]] ; then 
    echo "Gathering amdgpu compatible logs..."
    ts_amdgpu_compat_summary_logs
    ts_amdgpu_compat_full_logs
else
    echo "Unable to find either amdgpu or libgv on host system."
fi

echo $DOUBLE_BAR | tee -a $CONFIG_FILE_PLAT_INFO
echo LOG FILES: $CONFIG_PATH_PLAT_INFO:
tar -cvf $CONFIG_FILE_TAR $CONFIG_PATH_PLAT_INFO
tree $CONFIG_PATH_PLAT_INFO

exit 0

function host_guest_2() {

    # if on host. 

	if [[ -z `virt-what` ]] ; then
		dmesg | tee $CONFIG_FILE_DMESG_HOST
		cat /var/log/syslog | tee $CONFIG_FILE_SYSLOG_HOST
		cat  /var/log/kern.log | tee $CONFIG_FILE_KERN_LOG_HOST
        dmidecode | tee $CONFIG_FILE_DMIDECODE_HOST
        modinfo amdgpu | tee $CONFIG_FILE_MODINFO_AMDGPU_HOST
        for i in /sys/module/amdgpu/parameters/* ; do echo -n $i: ; cat $i ; done 2>&1 | tee $CONFIG_FILE_PARM_AMDGPU_HOST 
        rocminfo 2>&1 | tee $CONFIG_FILE_ROCMINFO_HOST
        rocm-smi --showall 2>&1 | tee $CONFIG_FILE_ROCMSMI_HOST
	else
		sshpass -p amd1234 ssh root@$vmIp 'dmesg' >  $CONFIG_FILE_DMESG_GUEST
		sshpass -p amd1234 ssh root@$vmIp 'cat /var/log/syslog' >  $CONFIG_FILE_SYSLOG_GUEST
		sshpass -p amd1234 ssh root@$vmIp 'cat /var/log/kern.log' >  $CONFIG_FILE_KERNLOG_GUEST
	fi
}

if [[ $p1 == "--help" ]] ; then
	clear
	echo "usage: "
	echo "$0 - get host information without specifying VM"
	echo "$0 <vm_index> get host and guest information. Use virsh to get vm index."
	exit 0
fi

apt install virt-what sshpass wcstools -y 

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
echo $DOUBLE_BAR | tee -a $CONFIG_FILE_PLAT_INFO

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
	echo "HOST: " | tee -a $CONFIG_FILE_PLAT_INFO
	echo $SINGLE_BAR | tee -a $CONFIG_FILE_PLAT_INFO
	echo "HOST IP:" | tee -a $CONFIG_FILE_PLAT_INFO
	echo `ifconfig | grep inet | grep -v inet6 | grep -v 127.0.0.1` | tee -a $CONFIG_FILE_PLAT_INFO
	echo $SINGLE_BAR | tee -a $CONFIG_FILE_PLAT_INFO
	echo "HOST OS:          "`lsb_release --all | grep -i description` | tee -a $CONFIG_FILE_PLAT_INFO
	echo $SINGLE_BAR | tee -a $CONFIG_FILE_PLAT_INFO
	echo "HOST KERNEL: 	"`uname -r` | tee -a $CONFIG_FILE_PLAT_INFO

	echo $SINGLE_BAR | tee -a $CONFIG_FILE_PLAT_INFO
	echo "HOST GPU: " | tee -a $CONFIG_FILE_PLAT_INFO
	cat /sys/bus/pci/drivers/gim/gpuinfo | egrep "Name|Bus" | tee -a $CONFIG_FILE_PLAT_INFO
	cat /sys/bus/pci/drivers/gim/gpubios | tee -a $CONFIG_FILE_PLAT_INFO
    echo "HOST GPUDRIVER:   "`lsmod | egrep "^amdkfd|^amdgpu"` | tee -a $CONFIG_FILE_PLAT_INFO
    echo $SINGLE_BAR | tee -a $CONFIG_FILE_PLAT_INFO
    echo "HOST AMDGPU:      "`modinfo amdgpu | egrep "^filename|^version"` | tee -a $CONFIG_FILE_PLAT_INFO
    echo $SINGLE_BAR | tee -a $CONFIG_FILE_PLAT_INFO
    echo "HOST GIM:         "`modinfo gim | egrep "^filename|^version"` | tee -a $CONFIG_FILE_PLAT_INFO

	host_guest_1

	#  ssh to vm, vm number is specified in $1.
	
	if [[ -z $1 ]] ; then
		echo $DOUBLE_BAR | tee -a $CONFIG_FILE_PLAT_INFO
		echo "VM No. is not specified. "
		echo $DOUBLE_BAR | tee -a $CONFIG_FILE_PLAT_INFO
		vmpIp=""
	else
		vmIp=`virsh domifaddr $p1 | egrep "[0-9]+\.[0-9]+\." | tr -s ' ' | cut -d ' ' -f5 | cut -d '/' -f1`
		if [[ -z $vmIp ]] ; then
			echo "Use virsh to determine running VM-s indices."
		exit 1
		fi
	fi
	host_guest_2
else
	echo "ERROR: Please run from host..." 
	exit 1
fi

# -----------------------------------------
# Get guest information
# -----------------------------------------

echo $DOUBLE_BAR | tee -a $CONFIG_FILE_PLAT_INFO
echo "VM IP:" 		$vmIp| tee -a $CONFIG_FILE_PLAT_INFO

if [[ -z $vmIp ]] ; then
	echo "vmIp is empty. Either failed to get address or did not specify vm index."
else

	echo $SINGLE_BAR | tee -a $CONFIG_FILE_PLAT_INFO
	echo "VM OS: 		"`sshpass -p amd1234 ssh root@$vmIp 'lsb_release --all | grep -i description'`| tee -a $CONFIG_FILE_PLAT_INFO
	echo $SINGLE_BAR | tee -a $CONFIG_FILE_PLAT_INFO
	echo "VM KERNEL:	"`sshpass -p amd1234 ssh root@$vmIp 'uname -r'` | tee -a $CONFIG_FILE_PLAT_INFO
	echo $SINGLE_BAR | tee -a $CONFIG_FILE_PLAT_INFO
	echo "VM HOSTNAME: 	"`sshpass -p amd1234 ssh root@$vmIp 'hostname'` | tee -a $CONFIG_FILE_PLAT_INFO
	echo $SINGLE_BAR | tee -a $CONFIG_FILE_PLAT_INFO
	echo "VM GPUDRIVER INFO:"`sshpass -p amd1234 ssh root@$vmIp 'modinfo amdgpu | egrep "^filename|^version"'` | tee -a $CONFIG_FILE_PLAT_INFO
	echo $SINGLE_BAR | tee -a $CONFIG_FILE_PLAT_INFO
	host_guest_2
	sshpass -p amd1234 ssh root@$vmIp 'modinfo amdgpu' >  $CONFIG_FILE_MODINFO_AMDGPU_GUEST
	echo $SINGLE_BAR | tee -a $CONFIG_FILE_PLAT_INFO
	sshpass -p amd1234 ssh root@$vmIp 'clinfo' >  $CONFIG_FILE_CLINFO_GUEST
	echo "CLINFO VERSION:" `sshpass -p amd1234 ssh root@$vmIp 'clinfo -v'` | tee -a  $CONFIG_FILE_PLAT_INFO
	echo $SINGLE_BAR | tee -a $CONFIG_FILE_PLAT_INFO
	host_guest_1
fi




