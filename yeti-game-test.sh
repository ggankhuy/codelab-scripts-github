function usage() {
	clear
	echo "Usage: "
	echo "$0 <game> <mode> <option> <terminal>"
	echo "where <game> is either 3dmark or doom"
	echo "where <mode>  is either yeti or linux"
	echo "where <options> is either 0 1 2"
	echo "0: - nostream"
	echo "1: - stream with 1 pc"
	echo "2: - stream with 2 pc"
	echo "where <terminal> is either t1, t2, client in case <options> is 2: stream with 2 pc
	exit 1
}
	
p1=$1
p2=$2	
p3=$3
p3=$4
mode=0		# 0 for yeti, 1 for linux
option=0
game=0

if [[ -z $p1 ]] || [[ -z $p2 ]]  || [[ -z $p3 ]]; then
	usage
fi 

if [[ $p1 == "3dmark" ]] ; then
	echo "3dmark is selected..."
	game=0
elif  [[ $p1 == "doom" ]] ; then
	echo "doom is selected..."
	game=1
else
	echo "Invalid game selected: $p1"
	exit 1
fi
if [[ $p2 == "linux" ]] ; then
	echo "linux option is not implemented yet. Sorry."
	exit 1
elif [[ $p2 == "yeti" ]] ; then
	echo "yeti mode is seleted."
	mode=0
else
	echo "invalid mode: $p2. Exiting..."
	exit 1
fi

if [[ $p3 -eq 2 ]] && [[ -z $p4 ]] ; then
	echo "You selected <option>=2 but then left <terminal type> empty. Exitin..."
	exit 1
fi

if [[ $p3 -eq  0 ]] ; then
	echo "no stream option is selected."
	option=0
elif  [[ $p3 -eq 1 ]] ; then
	echo "stream with 1 pc is selected."
	option=1
elif  [[ $p3 -eq 2 ]] ; then
	echo "stream with 2 pc is selected."
	option=2
else
	echo "Invalid option: $p3. Exiting..."
	exit 1
fi


if [[ $option -eq 0 ]] ; then
	if [[ $game -eq 0  ]] ; then
		clear
		echo setting up Yeti libraries...
		echo yeti 3dmark non-stream configuration run...
		sleep 3
		export LD_LIBRARY_PATH=~/yeti-eng-bundle/lib
		export VK_LOADER_DISABLE_YETI_EXT_WHITELIST=1
		
		#echo For render+discard mode:
		#source ~/yeti-eng-bundle/env/null.sh
		#echo NOTE: It seems that render+discard mode is broken with the latest eng bundle (20180830)
		
		echo For render+encode+discard:
		source ~/yeti-eng-bundle/env/vce_nostreamer.sh
		
		cd ~/yeti-content-bundle/3dmark/bin/yeti
		
		echo Run the 3dmark application the way you would for Linux XCB:
		./3dmark --asset_root=../../assets -i ../../configs/gt1.json
	elif [[ $game -eq 1 ]] ; then
		echo   doom does not support non-stream test option
		
	else
		echo "Invalid game: $game" 
		exit 1
	
elif [[ $option -eq 1 ]] ; then
	if [[ $game -eq 0 ]] ; then
		echo "Option 1 is partially implemented. Exiting..."
		exit 1
		clear
		echo setting up Yeti libraries...
		echo yeti 3dmark non-stream configuration run...
		sleep 3
		sudo uwf disable
		export LD_LIBRARY_PATH=~/yeti-eng-bundle/lib
		export VK_LOADER_DISABLE_YETI_EXT_WHITELIST=1
		
		echo Setup the swapchain for render+encode+stream:
		source ~/yeti-eng-bundle/env/vce.sh
		cd ~/yeti-content-bundle/3dmark/bin/yeti
		
		echo "Type, but do not execute the following command:"
		echo "./3dmark --asset_root=../../assets -i ../../configs/gt1.json"
		
		#NOTE: you can run a Yeti application with some debug output from the Vulkan loader and layers. To
		#do so, add VK_LOADER_DEBUG=all ahead of the application name. For example, for the 3dmark
		#command above, use:
		
		VK_LOADER_DEBUG=all ./3dmark --asset_root=../../assets -i ../../configs/gt1.json

	if [[ $game -eq 1 ]] ; then
		if [[ $p4 == "t1"  ]] ; then
			echo "Option 1 is partially implemented. Exiting..."
			exit 1
			export LD_LIBRARY_PATH=~/yeti-eng-bundle/lib
			export YETI_DISABLE_FABRICATED_CONNECTED=1
			source ~/yeti-eng-bundle/env/vce.sh
			cd ~/doom/yeti-release
			echo "Type, but do not execute the following command:"
			echo "./DOOM"
		elif [[ $p4 == "t2" ]] ; then
			pulseaudio --start			
			export LD_LIBRARY_PATH=~/yeti-eng-bundle/lib
			cd ~/yeti-eng-bundle/bin
			
			echo "Type, but do not execute the following command:"
			echo "./yeti_streamer -policy_config_file lan_policy.proto_ascii -connect_to_game_on_start -direct_webrtc -"
			echo "external_ip=127.0.0.1"
		elif [[ $p4 == "t3" ]] ; then
			export LD_LIBRARY_PATH=~/yeti-eng-bundle/lib
			cd ~/yeti-eng-bundle/bin
			echo "Type, but do not execute the following command:"
			echo "./game_client run-direct 127.0.0.1:44700"
			
		else
			echo "Unsupported terminal selection: $p4" ; exit 1
		fi 
	else
		echo "Unsupport game: $game" ; exit 1
	fi


elif [[ $option -eq 2 ]] ; then
	if [[ $game -eq 0 ]] ; then
		if [[ $p4 == "t1" ]] ; then
			clear
			echo setting up Yeti libraries...
			echo yeti 3dmark non-stream configuration run...
			echo terminal 1...
			sleep 3
			sudo uwf disable
			export LD_LIBRARY_PATH=~/yeti-eng-bundle/lib
			export VK_LOADER_DISABLE_YETI_EXT_WHITELIST=1
			
			echo Setup the swapchain for render+encode+stream:
			source ~/yeti-eng-bundle/env/vce.sh
			cd ~/yeti-content-bundle/3dmark/bin/yeti
			
			#NOTE: you can run a Yeti application with some debug output from the Vulkan loader and layers. To
			#do so, add VK_LOADER_DEBUG=all ahead of the application name. For example, for the 3dmark
			#command above, use:
			#VK_LOADER_DEBUG=all ./3dmark --asset_root=../../assets -i ../../configs/gt1.json
			
			echo Type, but do not execute the following command:
			echo ./3dmark --asset_root=../../assets -i ../../configs/gt1.json
		elif [[ $p4 == "t2" ]] ; then
			clear
			echo setting up Yeti libraries...
			echo yeti 3dmark non-stream configuration run...
			echo terminal 2...
			sleep 3
			pulseaudio --start
			export LD_LIBRARY_PATH=~/yeti-eng-bundle/lib
			cd ~/yeti-eng-bundle/bin
			
			echo "Type, but do not execute the following command:"
			echo "./yeti_streamer -policy_config_file lan_policy.proto_ascii -connect_to_game_on_start -direct_webrtc"
			echo "\-external_ip=\<IPv4 address of the Yeti computer\>"
		elif [[ $p4 == "client" ]] ; then
			clear
			echo setting up Yeti on client machine...
			
			apt install -y libc++abi-dev
			export LD_LIBRARY_PATH=~/yeti-eng-bundle/lib
			cd ~/yeti-eng-bundle/bin
			echo "Type, but do not execute the following command:"
			echo "./game_client run-direct <IPv4 address of the Yeti computer>:44700"
		else
			echo "Invalid  p3 is slipped through: $p4."
			exit 1
		fi
	elif [[ $game -eq 1 ]] ; then
		if [[ $p4 == "t1" ]] ; then
			export LD_LIBRARY_PATH=~/yeti-eng-bundle/lib
			export YETI_DISABLE_FABRICATED_CONNECTED=1		
			source ~/yeti-eng-bundle/env/vce.sh
			cd ~/doom
			echo "Type, but do not execute the following command"
			echo "./DOOM"
		elif [[ $p4 == "t2" ]] ; then
			pulseaudio --start
			export LD_LIBRARY_PATH=~/yeti-eng-bundle/lib
			cd ~/yeti-eng-bundle/bin
			echo "./yeti_streamer -policy_config_file lan_policy.proto_ascii -connect_to_game_on_start -direct_webrtc -"
			echo "external_ip=<IPv4 address of the Yeti computer>"
		elif [[ $p4 == "client" ]] ; then
			export LD_LIBRARY_PATH=~/yeti-eng-bundle/lib
			cd ~/yeti-eng-bundle/bin
			echo "Type, but do not execute the following command:"
			echo "./game_client run-direct <IPv4 address of the Yeti computer>:44700"
		else 
			echo "Invalid terminal selected: $p4 " ; exit 1
		fi
	else
		echo "Unsupported game: $game" ; exit 1
	fi
else
	echo "Invalid option is slipped through."
	exit 1
fi

