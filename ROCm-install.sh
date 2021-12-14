INSTALL_ROCM_SRC=1
INSTALL_ROCM_SRC_COPY=0
INSTALL_ROCM_SRC_COPY_INSTALL=0
CONFIG_FORCE_VERSION=1
CONFIG_VERSION=4.0
CONFIG_INTERVAL_SLEEP=60
CONFIG_DEBUG_BYPASS_INSTALLATION_ROCM=0

for var in "$@"
do
    if [[ ! -z `echo "$var" | grep "ip="` ]]  ; then
        echo "ip address: $var"
        p1=`echo $var | cut -d '=' -f2`
    fi

    if [[ ! -z `echo "$var" | grep "src="` ]]  ; then
        INSTALL_ROCM_SRC=`echo $var | cut -d '=' -f2`
    fi

    if [[ ! -z `echo "$var" | grep "sleep="` ]]  ; then
        CONFIG_INTERVAL_SLEEP=`echo $var | cut -d '=' -f2`
    fi

    if [[ ! -z `echo "$var" | grep "ver="` ]]  ; then
        CONFIG_VERSION=`echo $var | cut -d '=' -f2`
    fi
done

if [[ $p1 == '--help' ]] || [[ $p1 == "" ]]   ; then 
    echo "Usage: $0 <parameters>."
    echo "Parameters:"
    echo "ip=<IP_address>"
    echo "ver=<rocm version>"
    echo "src=<1=download rocm source, 0=do not download rocm source>. Default will download the source."
    echo "sleep=<wait time in seconds between reboots>"
    exit 0 ; 
fi

VM_IP=$p1

if [[ $CONFIG_DEBUG_BYPASS_INSTALLATION_ROCM -ne 1 ]] ; then
    for i in "apt install -y git python python3 tree net-tools" "apt remove amdgpu-dkms -y" "apt remove amdgpu-dkms-firmware -y" "apt update -y" "apt dist-upgrade -y" "apt install libnuma-dev -y " "echo rebooting ; sleep $CONFIG_INTERVAL_SLEEP ; reboot" ; do
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
else
    echo "Bypassing rocm installation for test purpose..."
fi

if [[ $INSTALL_ROCM_SRC -eq 1 ]] ; then
	if [[ $INSTALL_ROCM_SRC_COPY -eq 1 ]] ; then
        echo "Copying the rocm-source.sh script..."
		sshpass -p amd1234 scp ./rocm-source.sh root@$VM_IP:/root
        if [[ $? -ne 0 ]] ; then echo "failed to scp..." ; exit ; fi
		
		if [[ $INSTALL_ROCM_SRC_COPY_INSTALL -eq 1 ]] ; then
		        sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$VM_IP "cd ; ./rocm-source.sh $CONFIG_VERSION"
		fi
		
	else
		# following block not working and duplicate of rocm-source...
		echo "You specified to install rocm source. CONFIG_VERSION: $CONFIG_VERSION."
        sleep 10
		for i in "git config --global user.email \"you@example.com\"" \
            "git config --global user.name \"Your Name\"" \
            "git config --global color.ui false" \
			"mkdir -p ~/ROCm-$CONFIG_VERSION/" \
			"mkdir -p ~/bin/" \
			"echo 'cd ~/ROCm-$CONFIG_VERSION' >> ~/.bashrc" \
			"apt install curl -y && curl https://storage.googleapis.com/git-repo-downloads/repo > ~/bin/repo" \
			"chmod a+x ~/bin/repo" \
			"pwd" \
			"cd ~/ROCm-$CONFIG_VERSION ; ~/bin/repo init -u https://github.com/RadeonOpenCompute/ROCm.git -b roc-$CONFIG_VERSION.x ; ~/bin/repo sync"
		do
			echo -
			echo $i 
			echo - 
	        sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$VM_IP $i
            sleep 1
		done
		
	fi
else
	echo "Skipping rocm-source installation."
fi
sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$VM_IP  'ls -l /opt/'
sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$VM_IP  'sudo usermod -a -G video $LOGNAME ; cat /etc/group  | grep video'

