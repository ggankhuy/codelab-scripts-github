#  This script assumes all the VMs on either ixt39 or ixt70 is created afresh using autotest scripts i.e.
#  runtest 6 from autotest.
#  This script assumes the ens3 interface is used for streaming client and server. If the interface name is different
#  or absent, the result is unpredictable.
#  This script assumes the VM name is structured as debian-drop-<month>-<date>-debian-gpu<No>-vf<No> format 
#  using auto-test script. If the VM name is structured differently in any way, the result is unpredictable.
#  
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

CONFIG_IXT39_HOST_IP="10.216.66.51"
CONFIG_IXT70_HOST_IP="10.216.66.54"
CONFIG_HOST_IP=0
CONFIG_IXT38_GUEST_IP_START=(\
"10.216.66.67" \
"10.216.66.68" \
"10.216.66.69" \
"10.216.66.70")
CONFIG_IXT70_GUEST_IP_START=(\
"10.216.66.71" \
"10.216.66.72" \
"10.216.66.73" \
"10.216.66.74" \
"10.216.66.75" \
"10.216.66.76" \
"10.216.66.77" \
"10.216.66.78")

p1=$1

if [[ $p1 == "ixt39" ]] ; then
	CONFIG_HOST_IP=$CONFIG_IXT39_HOST_IP	
elif [[ $1 == "IXT70" ]] ; then
	CONFIG_HOST_IP=$CONFIG_IXT79_HOST_IP
else
	echo "ERROR: Invalid parameter."
	exit 1
fi

echo "HOST IP is set to: $CONFIG_HOST_IP"




