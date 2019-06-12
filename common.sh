SLEEP_TIME=1
GAME_3DMARK=0
GAME_DOOM=1
GAME_TR2=2
OPTION_EXTERNAL_IP=1
OPTION_LOCAL_IP=2
#REPO_SERVER_IP="10.217.74.231"
REPO_SERVER_IP="10.217.73.160"

game=0          # game
mode=0          # 0 for yeti, 1 for linux
option=0        # 0 for streaming, 1 and 2 for streaming with 1 or 2 pc respectively.

MODE_YETI=0
MODE_LINUX=1

OPTION_NOSTREAM=0
OPTION_STREAM_2PC=2

TERMINAL_T1=0
TERMINAL_T2=1
TERMINAL_CLIENT=2

SLEEP_TIME=1

#       Set either yeti or ggp  engineering bundle.

TR2_START_LOCATION=/usr/local/cloudcast/runit/

REPO_SERVER_IP="10.217.74.231"
#REPO_SERVER_IP="10.217.73.160"
REPO_SERVER_LOCATION=/repo/stadia

FILE_COPY_SCP=1
FILE_COPY_WGET=2
FILE_COPY_RSYNC=3
OPTION_FILE_COPY_PROTOCOL=$FILE_COPY_RSYNC

export DIR_YETI_CONTENT_BUNDLE=yeti-content-bundle
export DIR_GGP_ENG_BUNDLE=ggp-eng-bundle
export GGP_BUNDLE_VERSION=ggp-eng-bundle-20190413.tar.gz
export GGP_BUNDLE_VERSION=ggp-eng-bundle-20190518.tar.gz

#       Set either yeti or ggp  engineering bundle.

export DIR_ENG_BUNDLE_TO_USE=$DIR_GGP_ENG_BUNDLE
#export DIR_ENG_BUNDLE_TO_USE=$DIR_YETI_ENG_BUNDLE

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

#	
#	$1 - full path of file or directory to copy
#	$2 - destination  on local  file system to copy to.
	
function scp_robust ()
{
	HOST_SCP_SERVER_1=10.217.75.230
	HOST_SCP_SERVER_2=10.217.73.160
	HOST_SCP_SERVERS=(\
		$HOST_SCP_SERVER_1 \
		$HOST_SCP_SERVER_2 \
	)

	for i in ${HOST_SCP_SERVERS[@]}
	do
		echo "copying from $i..."
		#scp -C -v -o StrictHostKeyChecking=no -r root@$i:/$1 $2

                if [[ $OPTION_FILE_COPY_PROTOCOL == $FILE_COPY_RSYNC ]] ; then
	       	        sshpass -p amd1234 rsync -v -z -r -e "ssh -o StrictHostKeyChecking=no" root@$i:/$1 $2
                elif [[ $OPTION_FILE_COPY_PROTOCOL == $FILE_COPY_SCP ]] ; then
			scp -C -v -o StrictHostKeyChecking=no -r root@$i:/$1 $2
                else
                        echo "ERROR: Unknown or unsupported copy protocol."
                fi

		if [[ $? -eq 0 ]] ; then
			echo "Copy is successful."
			break
		else
			echo "Copy failed, trying next server."
		fi
	done
}
function setPathLdLibraryPath ()
{
        #export LD_LIBRARY_PATH=~/$DIR_ENG_BUNDLE_TO_USE/lib
        export LD_LIBRARY_PATH=/usr/local/cloudcast/lib

        if [[ -z `env | grep LD_LIBRARY_PATH` ]] ; then
                echo "it appears LD_LIBRARY_PATH env variable is not set up. Manually run:"
                #echo "export LD_LIBRARY_PATH=~/$DIR_ENG_BUNDLE_TO_USE/lib"
                echo "export LD_LIBRARY_PATH=/usr/local/cloudcast/lib"
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

	sleep $SLEEP_TIME

	# Setup ggp-eng-bundle in /usr/local/cloudcast.
	
	echo "Copying ggp-eng-bundle to /usr/local/cloudcast..."
	
	if [[ $OPTION_FILE_COPY_PROTOCOL == $FILE_COPY_RSYNC ]] ; then
        	sshpass -p amd1234 rsync -v -z -r -e "ssh -o StrictHostKeyChecking=no" root@$REPO_SERVER_IP:/$REPO_SERVER_LOCATION/florida/$GGP_BUNDLE_VERSION /tmp/
	elif [[ $OPTION_FILE_COPY_PROTOCOL == $FILE_COPY_SCP ]] ; then
        	sshpass -p amd1234 scp -C -v -o StrictHostKeyChecking=no -r root@$REPO_SERVER_IP:/$REPO_SERVER_LOCATION/florida/$GGP_BUNDLE_VERSION /tmp/
	else
        	echo "ERROR: Unknown or unsupported copy protocol."
	fi
	
	if [[ $? -ne 0 ]] ; then
        	echo "Failed to rsync copy ggp-eng-bundle"
        	exit 1
	fi

	mkdir -p /usr/local/cloudcast
	tar -xf /tmp/$GGP_BUNDLE_VERSION -C /usr/local/cloudcast --strip-components=1
	
	chmod -R a+rw /usr/local/cloudcast/
	mkdir /log
	chmod a+rw /log
	
	apt-get install freeglut3 pulseaudio libpulse-dev
	
	mkdir -p /opt/cloudcast/lib

	unlink /opt/cloudcast/lib/amdvlk64.so
	rm -rf /opt/cloudcast/lib/amdvlk64.so
	ln -s /opt/amdgpu-pro/lib/x86_64-linux-gnu/amdvlk64.so /opt/cloudcast/lib/amdvlk64.so
	mkdir -p ~/.local/share/vulkan/icd.d

	cp /usr/local/cloudcast/etc/vulkan/icd.d/ggpvlk.json ~/.local/share/vulkan/icd.d/
	mkdir -p /usr/local/cloudcast/etc/yetivlk
	cp /usr/local/cloudcast/etc/yetivlk/config.json /usr/local/cloudcast/etc/yetivlk

	echo "Soft links: "
	ls -l /usr/local/cloudcast/
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
	
	if [[ -z $2 ]] || [[ $2 -eq $OPTION_LOCAL_IP ]] ; then
		IP_TO_DISPLAY=127.0.0.1
	elif [[ $2 -eq $OPTION_EXTERNAL_IP ]]  ; then
		IP_TO_DISPLAY="$external_ip"
	fi

        echo "./dev/bin/yeti_streamer -policy_config_file dev/bin/lan_policy.proto_ascii -connect_to_game_on_start -direct_webrtc_ws -external_ip=$IP_TO_DISPLAY -port 44700 -null_audio=true"
}

#	Function used to process both terminal 1 (game itself) and terminal 2 (streaming server) from same shell window.
#	input: $1 - name of the game executable.
#	return: 1 - on any error.

function process_t1t2 ()
{
	GAME=$1

	DATE=`date +%Y%m%d-%H-%M-%S`
        LOG_DIR=/g/$DATE
        mkdir -p $LOG_DIR
        read -p "Press a key to start $GAME..."
        ./$GAME > $LOG_DIR/$GAME-$DATE.log &

        dhclient ens3

        if [[ $? -ne 0 ]] ; then
                echo "Warning: dhclient ens3 failed. ens3 interface might not have been able to get DHCP IP..."
        fi

        external_ip=`ifconfig ens3 | grep "inet " | tr -s " " | cut -d ' ' -f3`
        echo "external IP: " $external_ip

        if [[ -z $external_ip ]] ; then
                echo "Failed to get external IP: "  $external_ip
                exit 1
        fi

        sleep $SLEEP_TIME
        IP_TO_DISPLAY="$external_ip"
        cd /usr/local/cloudcast
        read -p "Press a key to start $GAME streaming server..."
        ./dev/bin/yeti_streamer \
                -policy_config_file dev/bin/lan_policy.proto_ascii \
                -connect_to_game_on_start -direct_webrtc_ws -external_ip=$IP_TO_DISPLAY \
                -port 44700 -null_audio=true > $LOG_DIR/TR2-stream-$DATE.log
}
