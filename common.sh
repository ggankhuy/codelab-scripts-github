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
        export LD_LIBRARY_PATH=~/yeti-eng-bundle/lib

        if [[ -z `env | grep LD_LIBRARY_PATH` ]] ; then
                echo "it appears LD_LIBRARY_PATH env variable is not set up. Manually run:"
                echo "export LD_LIBRARY_PATH=~/yeti-eng-bundle/lib"
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


function common_setup () {
	clear
	echo "Setup Yeti system for 3dmark on ubuntu 1604 / 1803..."
	
	DIR_YETI_ENG_BUNDLE=yeti-eng-bundle
	DIR_YETI_CONTENT_BUNDLE=yeti-content-bundle
	
	if [[ -z $GIB_DROP_ROOT ]] ; then
        	echo "GIB_DROP_ROOT is not defined. Please defined the root in ~/.bashrc"
        	exit 1
	fi

	if [[ ! -d ~/doom/yeti-release/ ]] ; then
		echo "~/doorm/pre-release is not created, creating now." 

		if [[ ! -d $GIB_DROP_ROOT/test-apps/Doom_Linux/ ]] ; then
			echo "Can not find DOOM source directory. Can not continue setup..."
			exit 1
		fi	
		mkdir -p ~/doom/yeti-release/

		echo "Copying doom now to ~/doom/yeti-release/"
		cp -vr $GIB_DROP_ROOT/test-apps/Doom_Linux/* ~/doom/yeti-release/
	fi		
	
	if [[ ! -d  $DIR_YETI_ENG_BUNDLE ]] ; then
        	echo "$DIR_YETI_ENG_BUNDLE does not exist yet, copying from $GIB_DROP_ROOT/test-apps/yeti..."
        	cp -vr $GIB_DROP_ROOT/test-apps/yeti/$DIR_YETI_ENG_BUNDLE ~
	else
        	echo "$DIR_YETI_ENG_BUNDLE already exist, skipping copy..."
	fi
	
	if [[ ! -d  $DIR_YETI_CONTENT_BUNDLE ]] ; then
        	echo "$DIR_YETI_CONTENT_BUNDLE does not exist yet, copying from $GIB_DROP_ROOT/test-apps/yeti..."
        	cp -vr $GIB_DROP_ROOT/test-apps/yeti/$DIR_YETI_CONTENT_BUNDLE ~
	else
        	echo "$DIR_YETI_CONTENT_BUNDLE already exist, skipping copy..."
	fi
	
	echo "Setup logging Needed for streaming configurations only – but do it now, so you don't forget:"
	mkdir -p /usr/local/cloudcast/log
	chmod -R a+rw /usr/local/cloudcast/
	ln -s ~/yeti-eng-bundle/lib/ /usr/local/cloudcast/lib
	mkdir /log
	chmod a+rw /log
	
	apt-get install freeglut3 pulseaudio libpulse-dev
	
	mkdir -p /opt/cloudcast/lib
	ln -s /opt/amdgpu-pro/lib/x86_64-linux-gnu/amdvlk64.so /opt/cloudcast/lib/amdvlk64.so
	mkdir -p /usr/local/cloudcast/
	ln -s ~/yeti-eng-bundle/lib /usr/local/cloudcast/lib
	mkdir -p ~/.local/share/vulkan/icd.d
	cp ~/yeti-eng-bundle/etc/vulkan/icd.d/yetivlk.json ~/.local/share/vulkan/icd.d/
	mkdir -p /usr/local/cloudcast/etc/yetivlk
	cp ~/yeti-eng-bundle/etc/yetivlk/config.json /usr/local/cloudcast/etc/yetivlk
	
}

function prompt_t2_with_ip () {
	echo "Type, but do not execute the following command:"
	echo "./yeti_streamer -policy_config_file lan_policy.proto_ascii -connect_to_game_on_start -direct_webrtc --console_stderr -external_ip=<ipv4>"
}
