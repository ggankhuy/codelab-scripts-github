if [[ -z '$1'  ]] ; then echo "Usage: $0 <ip_address> <1=for installing rocm source> of the server to which rocm to be installed." ; exit 1 ; fi

CONFIG_FORCE_VERSION=0
CONFIG_VERSION=4.0
CONFIG_INTERVAL_SLEEP=120

VM_IP=$1
INSTALL_ROCM_SRC=$2
INSTALL_ROCM_SRC_COPY=1
INSTALL_ROCM_SRC_COPY_INSTALL=0
for i in "apt remove amdgpu-dkms -y" "apt remove amdgpu-dkms-firmware -y" "apt update -y" "apt dist-upgrade -y" "apt install libnuma-dev -y " "echo rebooting ; sleep $CONFIG_INTERVAL_SLEEP ; reboot" ; do
        echo ----------
        echo $i
        echo ----------
        sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$VM_IP $i
done


sleep $CONFIG_INTERVAL_SLEEP
if [[ $CONFIG_FORCE_VERSION -eq 0 ]] ; then
	for i in "wget -qO - http://repo.radeon.com/rocm/apt/debian/rocm.gpg.key | sudo apt-key add -" \
		"echo 'cd ~/ROCm/' >> ~/.bashrc" \
	        "echo 'deb [arch=amd64] http://repo.radeon.com/rocm/apt/debian/ xenial main' | sudo tee /etc/apt/sources.list.d/rocm.list" \
	        "apt update" "apt install rocm-dkms -y" "modprobe amdgpu" "/opt/rocm/bin/rocminfo" "apt install clinfo -y" "clinfo"; 
	do
	        echo ----------
	        echo $i
	        echo ----------
	        sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$VM_IP $i
	done
else
	for i in "wget -qO - http://repo.radeon.com/rocm/apt/$CONFIG_VERSION/rocm.gpg.key | sudo apt-key add -" \
		"echo 'cd ~/ROCm/' >> ~/.bashrc" \
	        "echo 'deb [arch=amd64] http://repo.radeon.com/rocm/apt/$CONFIG_VERSION/ xenial main' | sudo tee /etc/apt/sources.list.d/rocm.list" \
	        "apt update" "apt install rocm-dkms -y" "modprobe amdgpu" "/opt/rocm/bin/rocminfo" "apt install clinfo -y" "clinfo"; 
	do
	        echo ----------
	        echo $i
	        echo ----------
	        sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$VM_IP $i
	done
fi
if [[ $INSTALL_ROCM_SRC -eq 1 ]] ; then
	if [[ $INSTALL_ROCM_SRC_COPY -eq 1 ]] ; then
		sshpass -p amd1234 scp ./rocm-source.sh root@$VM_IP:/root
		
		if [[ $INSTALL_ROCM_SRC_COPY_INSTALL -eq 1 ]] ; then
		        sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$VM_IP "cd ; ./rocm-source.sh"
		fi
		
	else
			# following block not working and duplicate of rocm-source...
		echo "You specified to install rocm source."
		for i in "apt install ocl-icd-opencl-dev libopenblas-dev  -y" \
			"mkdir -p ~/ROCm/" \
			"mkdir -p ~/bin/" \
			"echo 'cd ~/ROCm' >> ~/.bashrc" \
			"reboot"
		do
			echo -
			echo $i 
			echo - 
		        sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$VM_IP $i
		done
		
		sleep $CONFIG_INTERVAL_SLEEP
		for i in "apt install ocl-icd-opencl-dev libopenblas-dev  -y" \
			"apt install curl -y && curl https://storage.googleapis.com/git-repo-downloads/repo > ~/bin/repo" \
			"chmod a+x ~/bin/repo" \
			"pwd" \
			"sleep $CONFIG_INTERVAL_SLEEP" \
			"~/bin/repo init -u https://github.com/RadeonOpenCompute/ROCm.git -b roc-3.5.0"
		do
			echo -
			echo $i 
			echo - 
		        sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$VM_IP $i
		done
	fi
else
	echo "Skipping rocm-source installation."
fi
sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$VM_IP  'ls -l /opt/'
