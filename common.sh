SLEEP_TIME=1

function usage() {
        clear
        echo "Usage: "
        echo "$0 <game> <mode> <option> <terminal>"
        echo "where <game> is either 3dmark or doom"
        echo "where <game> is setup then perform initial setup for either 3dmark or doom."
        echo "where <mode>  is either yeti or linux"
        echo "where <options> is either 0 1 2"
        echo "0: - nostream"
        echo "1: - stream with 1 pc"
        echo "2: - stream with 2 pc"
        echo "where <terminal> is either t1, t2, client in case <options> is 2: stream with 2 pc"
        exit 1
}

function setPathLdLibraryPath ()
{
        export LD_LIBRARY_PATH=~/$DIR_YETI_ENG_BUNDLE/lib

        if [[ -z `env | grep LD_LIBRARY_PATH` ]] ; then
                echo "it appears LD_LIBRARY_PATH env variable is not set up. Manually run:"
                echo "export LD_LIBRARY_PATH=~/$DIR_YETI_ENG_BUNDLE/lib"
        fi
}

function setVkLoaderDisableYetiExtWhitelist ()
{
        export VK_LOADER_DISABLE_YETI_EXT_WHITELIST=1

        if [[ -z  `env | grep VK_LOADER_DISABLE_YETI_EXT_WHITELIST` ]] ; then
                echo "it appears VK_LOADER_DISABLE_YETI_EXT_WHITELIST env variable is not set up. Manually run:"
                echo "export VK_LOADER_DISABLE_YETI_EXT_WHITELIST=1"
        fi
}

function setYetiDisableFabricatedConnected () {
        export YETI_DISABLE_FABRICATED_CONNECTED=1

        if [[ -z  `env | grep YETI_DISABLE_FABRICATED_CONNECTED` ]] ; then
                echo "YETI_DISABLE_FABRICATED_CONNECTED env variable is not set up. Manually run:"
                echo "export YETI_DISABLE_FABRICATED_CONNECTED=1"
        fi
}

#       Display local IPv4

function displayIpv4 () {
        ipv4=`ifconfig | grep inet`
        echo $ipv4
}

function printBar () {
        echo "------------------------------------"
}

#	Check if the calling script is running inside VM
#	input: 	None
#	return  exit 1 if not running on VM.

function vm_check () {
#	Initialization function used by yeti-game-test.sh. 
#	input: $1 - GIB_DROP_ROOT location.
#	return: 1 - on any error.
#	r	0 - on success.	

	# Check if running in VM, if not, exit with error.

	apt install virt-what -y 

	if [[ -z `which virt-what` ]] ; then
		echo "Failed to install virt-what..."
		exit 1
	fi

	if [[ -z `virt-what` ]] ; then
		echo "virt-what returns null, it is not running inside vm: hostname: "
		echo `hostname`
		exit 1
	else
		echo "OK, running inside VM..."
	fi

	sleep $SLEEP_TIME
}

function common_setup () {
	clear
	echo "Setup Yeti system for 3dmark on ubuntu 1604 / 1803..."

	if [[ -z $1 ]] ; then
		echo "p1: $1 "
	else
		echo "Setting GIB_DROP_ROOT to $1..."
		export GIB_DROP_ROOT=$1
		GIB_DROP_ROOT=$1

		if [[ -z `cat ~/.bashrc | grep GIB_DROP_ROOT` ]] ; then
			echo "adding GIB_DROP_ROOT to bashrc..."
			echo "export GIB_DROP_ROOT=$1" >> ~/.bashrc
		else
			echo "GIB_DROP_ROOT already added to bash..."
		fi
	fi 

	if [[ -z `cat ~/.bashrc | grep "cd /git.co/ad-hog-scripts"` ]] ; then
		echo "adding: cd /git.co/ad-hoc-scripts..."
		echo "cd /git.co/ad-hoc-scripts" >> ~/.bashrc
	else
		echo "already present: cd /git.co/ad-hoc-scripts..."
	fi

	if [[ -z $GIB_DROP_ROOT ]] ; then
        	echo "GIB_DROP_ROOT is not defined. Please defined the root in ~/.bashrc"
        	exit 1
	fi

	DIR_YETI_ENG_BUNDLE=yeti-eng-bundle
	DIR_YETI_CONTENT_BUNDLE=yeti-content-bundle
	
	#  inserting amdgpu module just in case, sometimes not loaded.

	modprobe amdgpu
	modprobe amdkfd
	ret1=`lsmod | grep -u ^amdgpu`
	ret2=`lsmod | grep -u ^amdgpu`

	if [[ -z $ret1 ]]  || [[ -z $ret2 ]] ; then
		echo "Failed to install amdgpu or amdkfd (modprobe amdgpu/amdkfd), check the driver is installable or GPU is present."
		exit 1
		echo lsmod amdgpu: $ret1
		echo lsmod amdkfd: $ret2
	else
		echo lsmod amdgpu: $ret1
		echo lsmod amdkfd: $ret2
	fi
	
	sleep $SLEEP_TIME

	rm -rf ~/doom/
	mkdir -p ~/doom/

	echo "Setting up symlink for ~/doom/yeti-release/"
	#cp -vr $GIB_DROP_ROOT/test-apps/Doom_Linux/* ~/doom/yeti-release/
	ln -s $GIB_DROP_ROOT/test-apps/Doom_Linux/ ~/doom/yeti-release

	if [[ ! -d  $DIR_YETI_ENG_BUNDLE ]] ; then
        	echo "$DIR_YETI_ENG_BUNDLE does not exist yet, copying from $GIB_DROP_ROOT/test-apps/yeti..."
		unlink ~/$DIR_YETI_ENG_BUNDLE
		rm -rf ~/$DIR_YETI_ENG_BUNDLE
        	ln -s $GIB_DROP_ROOT/test-apps/yeti/$DIR_YETI_ENG_BUNDLE ~/$DIR_YETI_ENG_BUNDLE
	else
        	echo "$DIR_YETI_ENG_BUNDLE already exist, skipping copy..."
	fi
	
	if [[ ! -d  $DIR_YETI_CONTENT_BUNDLE ]] ; then
        	echo "$DIR_YETI_CONTENT_BUNDLE does not exist yet, copying from $GIB_DROP_ROOT/test-apps/yeti..."
		unlink ~/$DIR_YETI_CONTENT_BUNDLE
		rm -rf ~/$DIR_YETI_CONTENT_BUNDLE
        	ln -s $GIB_DROP_ROOT/test-apps/yeti/$DIR_YETI_CONTENT_BUNDLE ~/$DIR_YETI_CONTENT_BUNDLE
	else
        	echo "$DIR_YETI_CONTENT_BUNDLE already exist, skipping copy..."
	fi
	
	echo "Setup logging Needed for streaming configurations only – but do it now, so you don't forget:"
	mkdir -p /usr/local/cloudcast/log
	chmod -R a+rw /usr/local/cloudcast/
	unlink /usr/local/cloudcast/lib
	rm -rf /usr/local/cloudcast/lib
	ln -s ~/$DIR_YETI_ENG_BUNDLE/lib/ /usr/local/cloudcast/lib
	mkdir /log
	chmod a+rw /log
	
	apt-get install freeglut3 pulseaudio libpulse-dev
	
	mkdir -p /opt/cloudcast/lib

	unlink /opt/cloudcast/lib/amdvlk64.so
	rm -rf /opt/cloudcast/lib/amdvlk64.so
	ln -s /opt/amdgpu-pro/lib/x86_64-linux-gnu/amdvlk64.so /opt/cloudcast/lib/amdvlk64.so
	mkdir -p /usr/local/cloudcast/

	unlink /usr/local/cloudcast/lib
	rm -rf /usr/local/cloudcast/lib
	ln -s ~/$DIR_YETI_ENG_BUNDLE/lib /usr/local/cloudcast/lib
	mkdir -p ~/.local/share/vulkan/icd.d

	cp ~/$DIR_YETI_ENG_BUNDLE/etc/vulkan/icd.d/yetivlk.json ~/.local/share/vulkan/icd.d/
	mkdir -p /usr/local/cloudcast/etc/yetivlk
	cp ~/$DIR_YETI_ENG_BUNDLE/etc/yetivlk/config.json /usr/local/cloudcast/etc/yetivlk

	echo "Soft links: "
	ls -l ~/doom/
	ls -l /usr/local/cloudcast/
        ls -l ~/$DIR_YETI_ENG_BUNDLE
        ls -l ~/$DIR_YETI_CONTENT_BUNDLE
	ls -l /opt/cloudcast/lib/amdvlk64.so	
}

function prompt_t2_with_ip () {
	echo "Type, but do not execute the following command:"

	dhclient ens3
	
	if [[ $? -ne 0 ]] ; then
        	echo "Warning: dhclient ens3 failed. ens3 interface might not have been able to get DHCP IP..."
	fi
	
	external_ip=`ifconfig ens3 | grep "inet " | tr -s " " | cut -d ' ' -f3`
	echo "external IP: " $external_ip
	
	if [[ -z $external_ip ]] ; then
        	echo "Failed to get external IP: "  $external_ip
	fi
	
	sleep $SLEEP_TIME
	
	if [[ -z $1 ]] ; then
		echo "./yeti_streamer -policy_config_file lan_policy.proto_ascii -connect_to_game_on_start -direct_webrtc --console_stderr -external_ip=127.0.0.1"
	else
		#ifconfig | grep inet
		echo "./yeti_streamer -policy_config_file lan_policy.proto_ascii -connect_to_game_on_start -direct_webrtc --console_stderr -external_ip="$external_ip
	fi
}
