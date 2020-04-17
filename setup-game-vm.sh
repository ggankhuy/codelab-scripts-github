#  This script assumes all the VMs on either ixt39 or ixt70 is created afresh using autotest scripts i.e.
#  runtest 6 from autotest.
#  This script assumes the ens3 interface is used for streaming client and server. If the interface name is different
#  or absent, the result is unpredictable.
#  This script assumes the VM name is structured as debian-drop-<month>-<date>-debian-gpu<No>-vf<No> format 
#  using auto-test script. If the VM name is structured differently in any way, the result is unpredictable.
#  If there are VMs that are created for multiple drops, either running or shutdown, the result is extremely unpredictable. 
#  
#  Steps this tool takes:
#  Count all vms.
#  Load gim.
#  Start default network.
#  Turn on all vms.
#  Log on to each vm through ssh (determine ip using virsh domifaddr <vmno>
#  update /etc/network/interfaces with static ip from pool.
#  IP address range: 10.216.66.67-78.
#  Assignment:
#  
#  ixt39  4vm-s / 4 gpu-s, 10.216.66.67-70.
#  ixt70  8vm-s / 8 gpu-s, 10.216.66.71-78.

# Turn off all vm-s
# Set vm vcpu-s to 8 as standard.
# Turn on all VM-s 

source ./common.sh

DOUBLE_BAR="========================================================"
SINGLE_BAR="--------------------------------------------------------"
CONFIG_IXT39_HOST_IP="10.216.66.54"
CONFIG_IXT70_HOST_IP="10.216.66.51"
CONFIG_IXT21_HOST_IP="10.216.66.52"
CONFIG_IXT25_HOST_IP="10.216.66.53"
CONFIG_DAYTONAX1_HOST_IP="10.216.52.34"
CONFIG_DAYTONAX2_HOST_IP="10.216.52.30"
CONFIG_GB02_HOST_IP="10.216.52.62"
CONFIG_HOST_IP=0
CONFIG_GB02_IP_GUEST_IP_RANGE=(\
"0.0.0.0" \
"0.0.0.0" \
"0.0.0.0" \
"0.0.0.0")
CONFIG_DAYTONAX1_GUEST_IP_RANGE=(\
"0.0.0.0" \
"0.0.0.0" \
"0.0.0.0" \
"0.0.0.0")
CONFIG_DAYTONAX2_GUEST_IP_RANGE=(\
"0.0.0.0" \
"0.0.0.0" \
"0.0.0.0" \
"0.0.0.0")
CONFIG_IXT39_GUEST_IP_RANGE=(\
"10.216.66.67" \
"10.216.66.68" \
"10.216.66.69" \
"10.216.66.70")
CONFIG_IXT70_GUEST_IP_RANGE=(\
"10.216.66.71" \
"10.216.66.72" \
"10.216.66.73" \
"10.216.66.74" \
"10.216.66.75" \
"10.216.66.76" \
"10.216.66.77" \
"10.216.66.78")
CONFIG_VATS2_SUPPORT=1

CONFIG_GW="10.216.64.1"
CONFIG_DNS="10.216.64.5 10.218.15.1 10.218.15.2"
CONFIG_NETMASK="255.255.252.0"
CONFIG_SET_VCPUCOUNT=0
SETUP_GAME_VM_CLIENT=setup-game-vm-client.sh
DATE=`date +%Y%m%d-%H-%M-%S`

#	Following setting requires great diligence from user of this script. When running flag is set 
#	The  TOTAL_VMS will only count the running VM-s. This could be useful to not count non-running VM
#	which is irrelevant to current drop being worked on. That is because non-running VM could be left over
#	from other drop that was previosly worked on.

CONFIG_COUNT_ONLY_RUNNING_VM=1

#	If set, use the static ip assigned by QTS admin, otherwise use DHCP IP.

CONFIG_USE_STATIC_IP=0

# The loopback network interface
#auto lo
#iface lo inet loopback

#iface ens3 inet static
#address 10.216.66.78
#netmask 255.255.252.0
#network 10.216.64.0
#gateway 10.216.64.1
#dns-nameservers 10.216.64.5 10.218.15.1 10.218.15.2

DEBUG=1
VM_IPS=""
p2=$2
p1=$1

if [[ $p1 == "ixt39" ]] ; then
	CONFIG_HOST_IP=$CONFIG_IXT39_HOST_IP
	CONFIG_GUEST_IP_RANGE=(${CONFIG_IXT39_GUEST_IP_RANGE[@]})
elif [[ $1 == "ixt70" ]] ; then
	CONFIG_HOST_IP=$CONFIG_IXT70_HOST_IP
	CONFIG_GUEST_IP_RANGE=(${CONFIG_IXT70_GUEST_IP_RANGE[@]})
elif [[ $1 == "ixt21" ]] ; then
	CONFIG_HOST_IP=$CONFIG_IXT21_HOST_IP
elif [[ $1 == "ixt25" ]] ; then
	CONFIG_HOST_IP=$CONFIG_IXT25_HOST_IP
elif [[ $1 == "daytonax1" ]] ; then
	CONFIG_HOST_IP=$CONFIG_DAYTONAX1_HOST_IP
elif [[ $1 == "daytonax2" ]] ; then
	CONFIG_HOST_IP=$CONFIG_DAYTONAX2_HOST_IP
elif [[ $1 == "gb02" ]] ; then
	CONFIG_HOST_IP=$CONFIG_GB02_HOST_IP
else
	echo "ERROR: Invalid parameter."
	exit 1
fi

TOTAL_IPS=${#CONFIG_GUEST_IP_RANGE[@]}

echo "HOST IP is set to: $CONFIG_HOST_IP"
echo "GUEST IP RANGE is set to: $CONFIG_GUEST_IP_RANGE"
apt install sshpass -y

if [[ $? -ne 0 ]] ; then
	echo "ERROR. Failed to install sshpass package."
	echo "return code: $?"
	exit 1
fi

#  Count all vms.
#  
#  ixt39  4vm-s / 4 gpu-s, 10.216.66.67-70.

if [[ $CONFIG_VATS2_SUPPORT -eq 1 ]] ; then
	VM_GREP_PATTERN=vats
else
	VM_GREP_PATTERN=gpu
fi

TOTAL_VMS=`sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$CONFIG_HOST_IP "virsh list --all | grep -i $VM_GREP_PATTERN | grep running | wc -l"`

echo "TOTAL_VMS: $TOTAL_VMS"

if [[ $DEBUG -eq 1 ]] ; then
	echo ${#CONFIG_IXT39_GUEST_IP_RANGE[@]}
	echo ${#CONFIG_GUEST_IP_RANGE[@]}
fi

if  [[ $TOTAL_IPS -ne  $TOTAL_VMS ]] ; then
	echo "Error total VM found is not equal to CONFIG_GUEST_IP_RANGE."
	echo "Total VMs found: $TOTAL_VMS"
	echo "total IP-s: $TOTAL_IPS"
fi

#  Load gim.
#  Start default network.

sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$CONFIG_HOST_IP "modprobe gim"
sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$CONFIG_HOST_IP "virsh net-start default"

#   Set vCPUs to 8.

#  Turn on all vms.

for (( n=0; n < $TOTAL_VMS; n++ ))  ; do
	echo $DOUBLE_BAR
	echo n: $n
	GPU_INDEX=$n
	VM_INDEX=$(($n+1))
	echo "VM_INDEX: $VM_INDEX"
	
	VM_NAME=`sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$CONFIG_HOST_IP "virsh list --all | grep $VM_GREP_PATTERN | head -$(($GPU_INDEX+1)) | tail -1  | tr -s ' ' | cut -d ' ' -f3"`
	VM_NO=`sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$CONFIG_HOST_IP "virsh list --all | grep $VM_GREP_PATTERN | head -$(($GPU_INDEX)) | tail -1  | tr -s ' ' | cut -d ' ' -f2"`
	VM_IP=`virsh domifaddr $VM_NAME | grep ipv4 | tr -s ' ' | cut -d ' ' -f5 | cut -d '/' -f1`
	VM_IPS=`echo ${VM_IPS[@]} $VM_IP`
	echo VM_NAME: $VM_NAME, VM_INDEX: $VM_INDEX, VM_NO: $VM_NO, GPU_INDEX: $GPU_INDEX, VM_IP: $VM_IP
	sleep 1

	# Assign static ips now (halted development for now...)

	#CONFIG_IXT39_GUEST_IP_RANGE
	#CONFIG_IXT70_GUEST_IP_RANGE
	#CONFIG_GW="10.216.64.1"
	#CONFIG_DNS="10.216.64.5 10.218.15.1 10.218.15.2"
	#CONFIG_NETMASK="255.255.252.0"
	
	# The loopback network interface
	#auto lo
	#iface lo inet loopback
	
	#iface ens3 inet static
	#address 10.216.66.78
	#netmask 255.255.252.0
	#network 10.216.64.0
	#gateway 10.216.64.1
	#dns-nameservers 10.216.64.5 10.218.15.1 10.218.15.2
	
	#sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$CONFIG_HOST_IP "cat /etc/network/interfaces > /etc/network/interfaces.bak"

	# setup sshd and ssh client settings on guest VM-s.

	if [[ $2 == "ssh" ]] || [[ $2 == "" ]] ; then
		echo "adding user nonroot"
		sleep 3
		sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$VM_IP "adduser --disabled-password --gecos GECOS nonroot"	
		sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$VM_IP "echo -e \"amd1234\namd1234\n\" | passwd  nonroot"
		sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$VM_IP "usermod -aG sudo nonroot"	
		sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$VM_IP "apt install -y ssh-askpass ssh"	
	
		echo "setting ssh..."
		sleep 3
		sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$VM_IP "if [[ -z \`cat /etc/ssh/sshd_config | grep TCPKeepAlive\` ]] ; then echo TCPKeepAlive yes >> /etc/ssh/sshd_config ; fi;"
		sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$VM_IP "sed -i '/TCPKeepAlive/c \\TCPKeepAlive yes' /etc/ssh/sshd_config"
	
		sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$VM_IP "if [[ -z \`cat /etc/ssh/sshd_config | grep ClientAliveInterval\` ]] ; then echo ClientAliveInterval >> /etc/ssh/sshd_config ; fi;"
		sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$VM_IP "sed -i '/ClientAliveInterval/c \\ClientAliveInterval 60' /etc/ssh/sshd_config"
	
		sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$VM_IP "if [[ -z \`cat /etc/ssh/sshd_config | grep ClientAliveCountMax\` ]] ; then echo ClientAliveCountMax 10800 >> /etc/ssh/sshd_config ; fi;"
		sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$VM_IP "sed -i '/\\ClientAliveCountMax/c \\ClientAliveCountMax 10800' /etc/ssh/sshd_config"
	
		sshpass -p amd1234 rsync -v -z -r -e "ssh -o StrictHostKeyChecking=no" ./$SETUP_GAME_VM_CLIENT nonroot@$VM_IP:/home/nonroot/
		sshpass -p amd1234 ssh -o StrictHostKeyChecking=no nonroot@$VM_IP "nohup /home/nonroot/$SETUP_GAME_VM_CLIENT &"	
	fi

    if [[ -z ~/.ssh/id_rsa.pub ]] ; then
        echo "sshkey is not created."
    else
        echo "copying sshkey to VM, skipping due to bug..."
        #ssh-copy-id -i ~/.ssh/id_rsa.pub root@$VM_IP
    fi

	# collect dmesg only.

	if [[ $2 == "dmesg" ]]; then
		echo "Saving dmesg on VM$n..."
		mkdir -p /log/dmesg/$DATE
		sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$VM_IP "dmesg"	> /log/dmesg/$DATE/$p1.VM$n.dmesg.$DATE.log
	fi

	if [[ $2 == "dmesg-clear" ]]; then
		echo "Clearing dmesg on VM$n..."
		sleep 3
		sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$VM_IP "dmesg --clear"
		sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$VM_IP "dmesg | wc -l"
		echo "No. of lines in VM$n dmesg after clear: $lines"
	fi

	if [[ $CONFIG_SET_VCPUCOUNT -eq 1 ]] ; then
		echo "Turning off VM_NAME: $VM_NAME..."
		sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$CONFIG_HOST_IP "virsh shutdown $VM_NAME"
		echo "Done."	
	
		echo "Setting vCPUs to 8..."
		sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$CONFIG_HOST_IP "virsh setvcpus $VM_NAME 8 --config --maximum"
		sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$CONFIG_HOST_IP "virsh setvcpus $VM_NAME 8 --config"
		
		VCPU_COUNT=`sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$CONFIG_HOST_IP "virsh vcpucount $VM_NAME"`
		echo $VCPU_COUNT
		echo "Done."	
	
		if [[ $DEBUG -eq 1 ]] ; then
			echo "VM_NAME: $VM_NAME"
			echo "VM_NO: $VM_NO"		
		fi
	
		echo "Turning on VM_NAME: $VM_NAME..."
		sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$CONFIG_HOST_IP "virsh start $VM_NAME"
		sleep 30
		echo "Done."	
	fi 

done

#	Exit if p2 is dmesg,

if [[ $2 == "dmesg" ]] ; then
	if [[ $2 == "dmesg" ]]; then
		dmesg	> /log/dmesg/$DATE/$p1.host.dmesg.$DATE.log
	fi

	echo "dmesg for each VM is collected in /log/dmesg/$DATE."
	exit 0
fi

if [[ $2 == "dmesg-clear" ]]; then
	dmesg --clear
	lines=`dmesg | wc -l`
	echo "dmesg for host is cleared"
	echo "No. of dmesg line in host: $lines"
	exit 0
fi

TOTAL_VMS=`sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$CONFIG_HOST_IP "virsh list --all | grep -i gpu | wc -l"`

#  Log on to each vm through ssh (determine ip using virsh domifaddr <vmno>
#  update /etc/network/interfaces with static ip from pool.
#  IP address range: 10.216.66.67-78.
#  Assignment:

#echo -e "'amd1234b'\n'amd1234b'\n" | passwd  nonroot

echo "Finished copying $SETUP_GAME_VM_CLIENT. VM IP addresses:"
echo ${VM_IPS[@]}

echo "Setup /etc/ssh/sshd_config timeout settings on host."
sleep 5

apt install -y ssh

#sed -i '/TCPKeepAlive/c \\TCPKeepAlive yes' /etc/ssh/sshd_config
#if [[ -z `cat /etc/ssh/sshd_config | grep TCPKeepAlive` ]] ; then echo "TCPKeelAlive yes" >> /etc/ssh/sshd_config ; fi;

#sed -i '/ClientAliveInterval/c \\ClientAliveInterval 60' /etc/ssh/sshd_config
#if [[ -z `cat /etc/ssh/sshd_config | grep ClientAliveInterval` ]] ; then echo "ClientAliveInterval 60" >> /etc/ssh/sshd_config ; fi;

#sed -i '/ClientAliveCountMax/c \\ClientAliveCountMax 10800' /etc/ssh/sshd_config
#if [[ -z `cat /etc/ssh/sshd_config | grep ClientAliveCountMax` ]] ; then echo "ClientAliveCountMax 10800" >> /etc/ssh/sshd_config ; fi;

echo "Setup /etc/ssh/ssh_config timeout settings on host."
sleep 5

sed -i '/ServerAliveInterval/c \\ServerAliveInterval 60' /etc/ssh/ssh_config
if [[ -z `cat /etc/ssh/ssh_config | grep ServerAliveInterval` ]] ; then echo "ServerAliveInterval 60" >> /etc/ssh/ssh_config ; fi;

sed -i '/ServerAliveCountMax/c \\ServerAliveCountMax 10800' /etc/ssh/ssh_config
if [[ -z `cat /etc/ssh/ssh_config | grep ServerAliveCountMax` ]] ; then echo "ServerAliveCountMax 10800" >> /etc/ssh/ssh_config ; fi;
