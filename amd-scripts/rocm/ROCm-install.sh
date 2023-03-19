INSTALL_ROCM_SRC=1
INSTALL_ROCM_SRC_COPY=0
INSTALL_ROCM_SRC_COPY_INSTALL=0
CONFIG_FORCE_VERSION=1
CONFIG_VERSION=4.0
CONFIG_INTERVAL_SLEEP=60
CONFIG_DEBUG_BYPASS_INSTALLATION_ROCM=0

TYPE_OS_UBUNTU1804="1" # OK.
TYPE_OS_UBUNTU2004="2" # not supported yet.
TYPE_OS_CENTOS8="3" # tested on Centos8 Stream. 4.3 not working, intest.
TYPE_OS_RHEL7="4" # not supported yet.
TYPE_OS_SLES=3 # not supported yet.
PKG_EXEC=""

TYPE_OS=$TYPE_OS_UBUNTU1804
#TYPE_OS=$TYPE_OS_CENTOS8

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

OS_NAME=`sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$VM_IP \
"cat /etc/os-release  | grep ^NAME=  | tr -s ' ' | cut -d '\"' -f2"`
echo "OS_NAME: $OS_NAME"
sleep 3
case "$OS_NAME" in
   "Ubuntu")
      echo "Ubuntu is detected..."
      PKG_EXEC=apt
      ;;
   "CentOS Linux")
      echo "CentOS is detected..."
      PKG_EXEC=yum
      echo "Installing yum packages..." ; sleep 3
      yum install epel-release -y
      yum install sshpass -y
      ;;
   *)
     echo "Unsupported O/S, exiting..." ; exit 1
     ;;
esac

function install_rocm_ubuntu1804() {
	echo "Installing for Ubuntu1804."
    cmdArr=( "apt install -y git python3 tree net-tools" "apt remove amdgpu-dkms -y"  "apt remove amdgpu-dkms-firmware -y" "apt update -y" \
    "apt dist-upgrade -y" "apt install libnuma-dev -y" "reboot")
	if [[ $CONFIG_DEBUG_BYPASS_INSTALLATION_ROCM -ne 1 ]] ; then
        for i in "${cmdArr[@]}" ; do
 	            echo ---------- ; echo $i ; echo ---------- ; sleep 3 
	            sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$VM_IP $i
	    done

	    sleep $CONFIG_INTERVAL_SLEEP

        # Force version option is no longer necessary, always force since no force to latest versions since 4.3 does not work with 
        # this script anymore.

	    if [[ $CONFIG_FORCE_VERSION -eq 0 ]] ; then
            repoUrlBase="http://repo.radeon.com/rocm/apt/debian/"
	    else
            repoUrlBase="http://repo.radeon.com/rocm/apt/$CONFIG_VERSION/"
	    fi

        cmdArr=("wget -qO - $repoUrlBase/rocm.gpg.key | sudo apt-key add -" "echo 'cd ~/ROCm/' >> ~/.bashrc" \
            "echo 'deb [arch=amd64] $repoUrlBase xenial main' | sudo tee /etc/apt/sources.list.d/rocm.list" \
            "apt update" "apt install rocm-dkms -y" "modprobe amdgpu" "/opt/rocm/bin/rocminfo" "apt install clinfo -y" "clinfo")
    	for i in "${cmdArr[@]}" ; do
      	        echo ---------- ; echo issuing cmd: $i ; echo ---------- ; sleep 3
    	        sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$VM_IP $i
       	done
	else
	    echo "Bypassing rocm installation for test purpose..."
	fi
}

function install_rocm_centos8() {
	echo "Installing for Centos8."
    if [[ $CONFIG_DEBUG_BYPASS_INSTALLATION_ROCM -ne 1 ]] ; then
        if [[ $CONFIG_FORCE_VERSION -ne 0 ]] ; then
            repoUrlBase="https://repo.radeon.com/rocm/yum/$CONFIG_VERSION"
        else 
            repoUrlBase="https://repo.radeon.com/rocm/yum/rpm"
        fi

        cmdArrPrereq=("yum install git python -y")
        cmdArr=("sudo rpm --import https://repo.radeon.com/rocm/rocm.gpg.key" "sudo yum install -y epel-release" \
            "sudo yum install -y dkms kernel-headers-`uname -r` kernel-devel-`uname -r`" \
            "echo '[ROCm]' >  /etc/yum.repos.d/rocm.repo" \
            "echo 'name=ROCm' >>  /etc/yum.repos.d/rocm.repo" \
            "echo 'baseurl=$repoUrlBase' >>  /etc/yum.repos.d/rocm.repo" \
            "echo 'enabled=1' >>  /etc/yum.repos.d/rocm.repo" \
            "echo 'enabled=1' >>  /etc/yum.repos.d/rocm.repo" \
            "echo 'gpgcheck=1' >>  /etc/yum.repos.d/rocm.repo" \
            "echo 'gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key' >>  /etc/yum.repos.d/rocm.repo" \
            "yum install rocm-dkms" \
            "reboot")
        for i in "${cmdArrPrereq[@]}" ; do 
            echo ---------- ; echo $i ; echo ---------- ; sleep 3 ; sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$VM_IP $i 
        done
        for i in "${cmdArr[@]}" ; do 
            echo ---------- ; echo $i ; echo ---------- ; sleep 3 ; sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$VM_IP $i
        done

        echo "Sleeping for $CONFIG_INTERVAL_SLEEP while rebooting..."
        sleep $CONFIG_INTERVAL_SLEEP
        echo "Testing the installation..."

        cmdArr=("/opt/rocm/bin/rocminfo" "/opt/rocm/opencl/bin/clinfo")
        for i in "${cmdArr[@]}" ; do
            echo ---------- ; echo $i ; echo ---------- ; sleep 3
            sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$VM_IP $i
        done
    else
        echo "Bypassing rocm installation for test purpose..."
    fi
}

function install_src_common() {
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
            sleep 3
            cmdArr=("git config --global user.email \"you@example.com\"" \
                "git config --global user.name \"Your Name\"" \
                "git config --global color.ui false" \
                "mkdir -p ~/ROCm-$CONFIG_VERSION/" \
                "mkdir -p ~/bin/" \
                "echo 'cd ~/ROCm-$CONFIG_VERSION' >> ~/.bashrc" \
                "$PKG_EXEC install curl -y && curl https://storage.googleapis.com/git-repo-downloads/repo > ~/bin/repo" \
                "chmod a+x ~/bin/repo" \
                "pwd" \
                "cd ~/ROCm-$CONFIG_VERSION ; ~/bin/repo init -u https://github.com/RadeonOpenCompute/ROCm.git -b roc-$CONFIG_VERSION.x ; ~/bin/repo sync")

    		for i in "${cmdArr[@]}" ; do
    			echo - ; echo $i ; echo -  ; sleep 3
    	        sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$VM_IP $i
                sleep 1
    		done
    		
    	fi
    else
    	echo "Skipping rocm-source installation."
    fi
}
    
#   Start the installation.

case "$OS_NAME" in
   "Ubuntu")
      echo "Ubuntu is detected..."
      install_rocm_ubuntu1804
      ;;
   "CentOS Linux")
      echo "CentOS is detected..."
      install_rocm_centos8
      ;;
   *)
     echo "Unsupported O/S, exiting..." ; exit 1
     ;;
esac

install_src_common

sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$VM_IP  'ls -l /opt/'
sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$VM_IP  'sudo usermod -a -G video $LOGNAME ; cat /etc/group  | grep video'

