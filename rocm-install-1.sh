if [[ -z $1  ]] ; then echo "Usage: $0 <ip_address> <1=copy rocm source script to target" ; exit 1 ; fi

VM_IP=$1
INSTALL_ROCM_SRC=$2
INSTALL_ROCM_SRC_COPY=1
INSTALL_ROCM_SRC_COPY_RUN=0
WAIT_INTERVAL=10
for i in "apt remove amdgpu-dkms -y" "apt remove amdgpu-dkms-firmware -y" "apt update -y" "apt dist-upgrade -y" "apt install libnuma-dev -y " "echo rebooting ; sleep 15 ; reboot" ; do
        echo ----------
        echo $i
        echo ----------
        sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$VM_IP $i
done

sleep 15

timeout=0
for j in {0..20} ; do
        ping -c 2 $VM_IP
        if [[ $? -eq 0 ]]  ; then
		echo "Finished rebooting, can ping now..."
                break
        else
                echo "Can not ping, extending timeout..."
                fi
        timeout=$(($timeout+$WAIT_INTERVAL))
        echo "Wait time so far: $timeout seconds."

	if [[ $j -eq 20 ]] ; then
		echo "Timeout waiting for $VM_IP to become pingable."	
		exit 1
	fi
	sleep 10
done

if [[ -z $VM_IP ]] ; then
        echo "Failed to obtain IP, , too unsafe to continue...."
        exit 1
fi

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

if [[ $INSTALL_ROCM_SRC -eq 1 ]] ; then
	if [[ $INSTALL_ROCM_SRC_COPY -eq 1 ]] ; then
		sshpass -p amd1234 scp ./rocm-source.sh root@$VM_IP:/root
		
		if [[ $INSTALL_ROCM_SRC_COPY_RUN -eq 1 ]] ; then
		        sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$VM_IP "cd ; ./rocm-source.sh"
		fi
		
	else
		echo "You specified to install rocm source. Installing default version: 3.5.0..."
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
		
		sleep 15
		for i in "apt install ocl-icd-opencl-dev libopenblas-dev  -y" \
			"apt install curl -y && curl https://storage.googleapis.com/git-repo-downloads/repo > ~/bin/repo" \
			"chmod a+x ~/bin/repo" \
			"pwd" \
			"sleep 5" \
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
