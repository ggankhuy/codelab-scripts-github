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
CONFIG_HOST_IP=0
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

CONFIG_GW="10.216.64.1"
CONFIG_DNS="10.216.64.5 10.218.15.1 10.218.15.2"
CONFIG_NETMASK="255.255.252.0"

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

p1=$1

if [[ $p1 == "ixt39" ]] ; then
	CONFIG_HOST_IP=$CONFIG_IXT39_HOST_IP
	CONFIG_GUEST_IP_RANGE=(${CONFIG_IXT39_GUEST_IP_RANGE[@]})
elif [[ $1 == "ixt70" ]] ; then
	CONFIG_HOST_IP=$CONFIG_IXT70_HOST_IP
	CONFIG_GUEST_IP_RANGE=(${CONFIG_IXT70_GUEST_IP_RANGE[@]})
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
TOTAL_VMS=`sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$CONFIG_HOST_IP "virsh list --all | grep -i gpu | wc -l"`
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
	VM_INDEX=$(($n+1))
	echo "VM_INDEX: $VM_INDEX"
	
	#VM_NAME=`sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$CONFIG_HOST_IP "virsh list --all | grep gpu | head -$(($VM_INDEX+1)) | tail -1  | tr -s ' ' | cut -d ' ' -f3"`
	VM_NAME=`sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$CONFIG_HOST_IP "virsh list --all | grep gpu$n | tr -s ' ' | cut -d ' ' -f3"`
	sleep $SLEEP_TIME_2
	#VM_NO=`sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$CONFIG_HOST_IP "virsh list --all | grep gpu | head -$((n+1)) | tail -1  | tr -s ' ' | cut -d ' ' -f2"`
	VM_NO=`sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$CONFIG_HOST_IP "virsh list --all | grep gpu$n | tr -s ' ' | cut -d ' ' -f2"`
	sleep $SLEEP_TIME_2

	echo "Turning off VM_NAME: $VM_NAME..."
	sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$CONFIG_HOST_IP "virsh shutdown $VM_NAME"
	echo "Done."	

	echo "Setting vCPUs to 8..."
	sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$CONFIG_HOST_IP "virsh setvcpus $VM_NAME 8 --config --maximum"
	sleep $SLEEP_TIME_2

	sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$CONFIG_HOST_IP "virsh setvcpus $VM_NAME 8 --config"
	sleep $SLEEP_TIME_2
	
	VCPU_COUNT=`sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$CONFIG_HOST_IP "virsh vcpucount $VM_NAME"`
	echo $VCPU_COUNT
	echo "Done."	

	if [[ $DEBUG -eq 1 ]] ; then
		echo "VM_NAME: $VM_NAME"
		echo "VM_NO: $VM_NO"		
	fi

	echo "Turning on VM_NAME: $VM_NAME..."
	sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$CONFIG_HOST_IP "virsh start $VM_NAME"
	sleep $SLEEP_TIME_2
	echo "Done."	

	# Assign static ips now

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

	sleep $SLEEP_TIME_2
done

TOTAL_VMS=`sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$CONFIG_HOST_IP "virsh list --all | grep -i gpu | wc -l"`

#  Log on to each vm through ssh (determine ip using virsh domifaddr <vmno>
#  update /etc/network/interfaces with static ip from pool.
#  IP address range: 10.216.66.67-78.
#  Assignment:
