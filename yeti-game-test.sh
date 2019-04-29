#	Launcher scripts for 3DMARK and DOOM.
#	To run the script simply provide root location of drop package in ~./bashrc
#	This script has been tested with root account only and assumes everything is copied to root folder.
#	Non-root account has not been tested.
#	i.e. in ~/.bashrc:
#	export GIB_DROP_ROOT=/work/drop-March-21-debian/

#	After that ./yeti-game-setup.sh setup to run preliminary setup for either 3dmark or DOOM.
#	After that from each of terminal 1,2 and 3 (client) 
#	. ./yeti-game-setup doom yeti 2 t1
#	. ./yeti-game-setup doom yeti 2 t2
#	. ./yeti-game-setup doom yeti 2 client
#	At the end of each run, the script will prompt with the last syntax to run the actual game for each terminals
#	with seconds apart for easier copy and paste and launch (type but not run).

#	print bar.

function printBar () {
	echo "------------------------------------"
}

#	Prints usage information.

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

#	Display local IPv4

function displayIpv4 () {
	ipv4=`ifconfig | grep inet`
	echo $ipv4
}

#	Set paths

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

#	Actual scripts starts here.

p1=$1
p2=$2	
p3=$3
p4=$4

game=0		# game
mode=0		# 0 for yeti, 1 for linux
option=0	# 0 for streaming, 1 and 2 for streaming with 1 or 2 pc respectively.

GAME_3DMARK=0
GAME_DOOM=1

MODE_YETI=0
MODE_LINUX=1

OPTION_NOSTREAM=0
OPTION_STREAM_1PC=1
OPTION_STREAM_2PC=2

TERMINAL_T1=0
TERMINAL_T2=1
TERMINAL_CLIENT=2

if [[ ! -d ~/yeti-eng-bundle ]] ; then
	clear
	printBar
	echo "This script assumes the yeti-eng-bundle is on ~. "
	printBar
	exit 1
fi 
#	Process help request. 

if [[ $1 == "--help"  ]] || [[ -z $1 ]] ; then
	usage
	exit 1
fi

#	p1 is for either 3dmark or doom (unless more support added.

if [[ $p1 == "3dmark" ]] ; then
	echo "3dmark is selected..."
	game=$GAME_3DMARK
elif  [[ $p1 == "doom" ]] ; then
	echo "doom is selected..."
	game=$GAME_DOOM
elif [[ $p1 == "setup" ]] ; then
	source ./common.sh
	echo "setting up the system for test."
	common_setup
	exit 0
else
	echo "Invalid game selected: $p1"
	exit 1
fi

if [[ -z $2  ]] || [[ -z $3 ]] || [[ -z $4 ]] ; then
	echo "p0 selected is for game not setup. Therefore you need to supply p2-4 parameters."
	usage
	exit 1
fi
#	p2 is for either linux or yeti.

if [[ $p2 == "linux" ]] ; then
	echo "linux option is not implemented yet. Sorry."
	mode=$MODE_LINUX
	exit 1
elif [[ $p2 == "yeti" ]] ; then
	echo "yeti mode is seleted."
	mode=$MODE_YETI
else
	echo "invalid mode: $p2. Exiting..."
	exit 1
fi

#	p3 is for  no-stream, stream 1 pc or 2 pc option.
#	With no stream 		p3=0, terminal is not needed to be specified in p4.
#	With stream 1 pc	p3=1, terminal need to be specified with either p4=t1, t2 or client.
#	With stream 2 pc	p3=2, terminal need to be specified with either p4=t1, t2 or client.

if [[ $p3 -eq 2 ]] && [[ -z $p4 ]] ; then
	echo "You selected <option>=2 but then left <terminal type> empty. Exiting.."
	exit 1
fi

if [[ $p3 -eq  0 ]] ; then
	echo "no stream option is selected."
	option=$OPTION_NO_STREAM
elif  [[ $p3 -eq 1 ]] ; then
	echo "stream with 1 pc is selected."
	option=$OPTION_STREAM_1PC
elif  [[ $p3 -eq 2 ]] ; then
	echo "stream with 2 pc is selected."
	option=$OPTION_STREAM_2PC
else
	echo "Invalid option: $p3. Exiting..."
	exit 1
fi

if [[ $option -eq $OPTION_NOSTREAM ]] ; then
	if [[ $game -eq $GAME_3DMARK ]] ; then
		clear
		echo setting up Yeti libraries...
		echo yeti 3dmark non-stream configuration run...
		sleep 3

		setPathLdLibraryPath
		setVkLoaderDisableYetiExtWhitelist
		
		#echo For render+discard mode:
		#source ~/yeti-eng-bundle/env/null.sh
		#echo NOTE: It seems that render+discard mode is broken with the latest eng bundle (20180830)
		
		echo For render+encode+discard:
		source ~/yeti-eng-bundle/env/vce_nostreamer.sh
		
		cd ~/yeti-content-bundle/3dmark/bin/yeti
		
		echo Run the 3dmark application the way you would for Linux XCB:
		./3dmark --asset_root=../../assets -i ../../configs/gt1.json
	elif [[ $game -eq $GAME_DOOM ]] ; then
		echo Doom does not support non-stream test option.
		
	else
		echo "Invalid game: $game" 
		exit 1
	fi

	
elif [[ $option -eq $OPTION_STREAM_1PC ]] ; then
	echo "OPTION: STREAM 1PC ..." ; sleep 2
	if [[ $game -eq $GAME_3DMARK ]] ; then
		echo "GAME: 3DMARK ..." ; sleep 2
		if [[ $p4 == "t1"  ]] ; then
			echo "Terminal1." ; sleep 2
			sudo uwf disable
			setPathLdLibraryPath
			setVkLoaderDisableYetiExtWhitelist
			echo Setup the swapchain for render+encode+stream:
			source ~/yeti-eng-bundle/env/vce.sh
			cd ~/yeti-content-bundle/3dmark/bin/yeti
			
			echo "Type, but do not execute the following command:"
			echo "./3dmark --asset_root=../../assets -i ../../configs/gt1.json"
			
			#NOTE: you can run a Yeti application with some debug output from the Vulkan loader and layers. To
			#do so, add VK_LOADER_DEBUG=all ahead of the application name. For example, for the 3dmark
			#command above, use:
			#VK_LOADER_DEBUG=all ./3dmark --asset_root=../../assets -i ../../configs/gt1.json
		elif [[ $p4 == "t2" ]] ; then
			echo "Terminal2". ; sleep 2
			clear
			echo setting up Yeti libraries...
			echo yeti 3dmark non-stream configuration run...
			pulseaudio --start

			setPathLdLibraryPath
			cd ~/yeti-eng-bundle/bin
			echo "Type, but do not execute the following command:"
			echo "./yeti_streamer -policy_config_file lan_policy.proto_ascii -connect_to_game_on_start -direct_webrtc -external_ip=127.0.0.1"
		elif [[ $p4 == "client" ]] ; then
			echo "Terminal3 / client"; sleep 2
			clear
			echo setting up Yeti on client machine...
			
			apt install -y libc++abi-dev

			setPathLdLibraryPath

			cd ~/yeti-eng-bundle/bin
			echo "Type, but do not execute the following command:"
			echo "./game_client run-direct <IPv4 address of the Yeti computer>:44700"
		else
			echo "Invalid  p4 is slipped through: $p4."
			exit 1
		fi	

	elif [[ $game -eq $GAME_DOOM ]] ; then
		echo "GAME: DOOM..." ; sleep 2
		if [[ $p4 == "t1"  ]] ; then
			echo "Terminal1." ; sleep 2
			setPathLdLibraryPath
			setVkLoaderDisableYetiExtWhitelist

			source ~/yeti-eng-bundle/env/vce.sh
			mkdir -p ~/doom/yeti-release
			cd ~/doom/yeti-release
			echo "Type, but do not execute the following command from this directory ~/doom/yeti-release:"
			echo "./DOOM"
		elif [[ $p4 == "t2" ]] ; then
			echo "Terminal2." ; sleep 2
			pulseaudio --start			

			setPathLdLibraryPath

			cd ~/yeti-eng-bundle/bin
			
			echo "Type, but do not execute the following command:"
			echo "./yeti_streamer -policy_config_file lan_policy.proto_ascii -connect_to_game_on_start -direct_webrtc -"
			echo "external_ip=127.0.0.1"
		elif [[ $p4 == "client" ]] ; then
			echo "Terminal3." ; sleep 2
			clear
			echo setting up Yeti on client machine...
			apt install -y libc++abi-dev
			setPathLdLibraryPath
			cd ~/yeti-eng-bundle/bin
			echo "Type, but do not execute the following command:"
			echo "./game_client run-direct 127.0.0.1:44700"
			
		else
			echo "Unsupported terminal selection: $p4" ; exit 1
		fi 
	else
		echo "Unsupport game: $game" ; exit 1
	fi
elif [[ $option -eq $OPTION_STREAM_2PC ]] ; then
	echo "OPTION: STREAM 2 PC." ; sleep 2

	if [[ $game -eq $GAME_3DMARK ]] ; then
		echo "GAME: 3DMARK." ; sleep 2
		if [[ $p4 == "t1" ]] ; then
			echo "Terminal1." ; sleep 2
			clear
			echo setting up Yeti libraries...
			echo yeti 3dmark non-stream configuration run...
			echo terminal 1...
			sleep 3
			sudo uwf disable

			setPathLdLibraryPath
			setVkLoaderDisableYetiExtWhitelist
			#setYetiDisableFabricatedConnected

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
			echo "Terminal2." ; sleep 2
			clear
			echo setting up Yeti libraries...
			echo yeti 3dmark non-stream configuration run...
			echo terminal 2...
			sleep 3
			pulseaudio --start

			setPathLdLibraryPath
			cd ~/yeti-eng-bundle/bin
			displayIpv4
			echo "Type, but do not execute the following command:"
			echo "./yeti_streamer -policy_config_file lan_policy.proto_ascii -connect_to_game_on_start -direct_webrtc -external_ip=<ipv4>"

		elif [[ $p4 == "client" ]] ; then
			echo "Terminal3 / client." ; sleep 2
			clear
			echo setting up Yeti on client machine...
			
			apt install -y libc++abi-dev

			setPathLdLibraryPath

			cd ~/yeti-eng-bundle/bin
			echo "Type, but do not execute the following command:"
			echo "./game_client run-direct <IPv4 address of the Yeti computer>:44700"
		else
			echo "Invalid  p4 is slipped through: $p4."
			exit 1
		fi
	elif [[ $game -eq $GAME_DOOM ]] ; then
		echo "GAME: DOOM" ; sleep 2
		if [[ $p4 == "t1" ]] ; then			
			echo "Terminal1." ; sleep 2

			if [[ ! "$(ls -A ~/yeti-eng-bundle)" ]] ; then
    			echo "<path> is empty!"
			else
    			echo "<path> is not empty"
			fi

			setPathLdLibraryPath
			setYetiDisableFabricatedConnected

			source ~/yeti-eng-bundle/env/vce.sh
			mkdir -p ~/doom/yeti-release
			cd ~/doom/yeti-release

                        if [[ ! -f ~/doom/yeti-release/DOOM ]] ; then
                                echo "the DOOM is not in ~/doom/yeti-release, copy it first! "
                                exit 1
                        fi

			echo "If after executing this script, you are in this directory and then manually go to this directory: ~/doom/yeti-release"
			echo "Type, but do not execute the following command"
			echo "./DOOM"
		elif [[ $p4 == "t2" ]] ; then
			echo "Terminal2." ; sleep 2
			pulseaudio --start

			if [[ $? != 0 ]] ; then 
				echo "Failed to run pulseaudio, does it exist>"
				exit 1
			fi
			
			setPathLdLibraryPath

			cd ~/yeti-eng-bundle/bin

			if [[ $? != 0 ]] ; then 
				echo "Failed to run pulseaudio, does it exist>"
				exit 1
			fi

			displayIpv4
			echo "Type, but do not execute the following command:"
			echo "./yeti_streamer -policy_config_file lan_policy.proto_ascii -connect_to_game_on_start -direct_webrtc -external_ip=<ipv4>"

		elif [[ $p4 == "client" ]] ; then
			echo "Terminal3 / client." ; sleep 2

			export LD_LIBRARY_PATH=~/yeti-eng-bundle/lib

			setPathLdLibraryPath

			cd ~/yeti-eng-bundle/bin

			if [[ $? -ne 0 ]] ; then
				echo "Can not cd into ~/yet-end-bundle/bin"
				exit 1
			fi
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

