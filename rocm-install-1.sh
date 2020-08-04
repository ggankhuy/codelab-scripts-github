if [[ -z '$1'  ]] ; then echo "Usage: $0 <ip_address> <1=for installing rocm source> of the server to which rocm to be installed." ; exit 1 ; fi

VM_IP=$1
INSTALL_ROCM_SRC=$2

for i in "apt remove amdgpu-dkms -y" "apt update -y" "apt dist-upgrade -y" "apt install libnuma-dev -y " "echo rebooting ; sleep 15 ; reboot" ; do
        echo ----------
        echo $i
        echo ----------
        sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$VM_IP $i
done

sleep 15
for i in "wget -qO - http://repo.radeon.com/rocm/apt/debian/rocm.gpg.key | sudo apt-key add -" \
        "echo 'deb [arch=amd64] http://repo.radeon.com/rocm/apt/debian/ xenial main' | sudo tee /etc/apt/sources.list.d/rocm.list" \
        "apt update" "apt install rocm-dkms -y" "modprobe amdgpu" "/opt/rocm/bin/rocminfo" "apt install clinfo -y" "clinfo"; 
do
        echo ----------
        echo $i
        echo ----------
        sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$VM_IP $i
done

if [[ $INSTALL_ROCM_SRC -eq 1 ]] ; then
	echo "You specified to install rocm source. Installing default version: 3.5.0..."
	for i in "apt install ocl-icd-opencl-dev libopenblas-dev  -y" \
		"mkdir -p ~/ROCm/" \
		"pushd  ~/ROCm/" \
		"mkdir -p ~/bin/" \
		"apt install curl -y && curl https://storage.googleapis.com/git-repo-downloads/repo > ~/bin/repo" \
		"chmod a+x ~/bin/repo" \
		"~/bin/repo init -u https://github.com/RadeonOpenCompute/ROCm.git -b roc-3.5.0" \
		"~/bin/repo sync" \
		"echo 'ROCm source is downloaded to ~/ROCm'" \
		"echo 'push ~/ROCm to get there...'"\ ; do
		echo -
		echo $i 
		echo - 
	        sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$VM_IP $i
	done
else
	echo "Skipping rocm-source installation."
fi
