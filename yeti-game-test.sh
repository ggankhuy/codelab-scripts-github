#	Launcher scripts for 3DMARK, DOOM and TR2.
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
#	Prints usage information.

source ./common.sh

#	Actual scripts starts here.

p1=$1
p2=$2	
p3=$3
p4=$4

game=0		# game
mode=0		# 0 for yeti, 1 for linux
option=0	# 0 for streaming, 1 and 2 for streaming with 1 or 2 pc respectively.

MODE_YETI=0
MODE_LINUX=1

OPTION_NOSTREAM=0
OPTION_STREAM_1PC=1
OPTION_STREAM_2PC=2

TERMINAL_T1=0
TERMINAL_T2=1
TERMINAL_CLIENT=2

SLEEP_TIME=1

DIR_YETI_ENG_BUNDLE=yeti-eng-bundle
DIR_YETI_CONTENT_BUNDLE=yeti-content-bundle
DIR_GGP_ENG_BUNDLE=ggp-eng-bundle

#	Set either yeti or ggp  engineering bundle.

#DIR_ENG_BUNDLE_TO_USE=$DIR_GGP_ENG_BUNDLE
DIR_ENG_BUNDLE_TO_USE=$DIR_YETI_ENG_BUNDLE

TR2_START_LOCATION=/usr/local/cloudcast/runit/

REPO_SERVER_IP="10.217.74.231"
#REPO_SERVER_IP="10.217.73.160"
REPO_SERVER_LOCATION=/repo/stadia


FILE_COPY_SCP=1
FILE_COPY_WGET=2
OPTION_FILE_COPY_PROTOCOL=$FILE_COPY_WGET

vm_check
sleep $SLEEP_TIME

#	apt packages 

apt install sshpass -y

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
elif [[ $p1 == "tr2" ]] ; then
	echo "tr2 is selected..."
	game=$GAME_TR2
elif [[ $p1 == "setup" ]] ; then
	echo "setting up the system for test."
	echo "p2: $p2..."
	common_setup $p2
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

if [[ -z $DIR_ENG_BUNDLE_TO_USE ]] ; then
	echo "ERROR: DIR_ENG_BUNDLE_TO_USE is not defined: $DIR_ENG_BUNDLE_TO_USE"
	exit 1
else
	echo "OK, DIR_ENG_BUNDLE_TO_USE: $DIR_ENG_BUNDLE_TO_USE"
fi

if [[ -z $DIR_YETI_CONTENT_BUNDLE ]] ; then
	echo "ERROR: DIR_YETI_CONTENT_BUNDLE is not defined: $DIR_YETI_CONTENT_BUNDLE"
	exit 1
else
	echo "OK, DIR_YETI_CONTENT_BUNDLE: $DIR_YETI_CONTENT_BUNDLE"
fi

if [[ $option -eq $OPTION_NOSTREAM ]] ; then
	if [[ $game -eq $GAME_3DMARK ]] ; then
		clear
		echo setting up Yeti libraries...
		echo yeti 3dmark non-stream configuration run...
		sleep $SLEEP_TIME

		setPathLdLibraryPath
		setVkLoaderDisableYetiExtWhitelist
		
		#echo For render+discard mode:
		#source ~/$DIR_ENG_BUNDLE_TO_USE/env/null.sh
		#echo NOTE: It seems that render+discard mode is broken with the latest eng bundle (20180830)
		
		echo For render+encode+discard:
		source ~/$DIR_ENG_BUNDLE_TO_USE/env/vce_nostreamer.sh
		
		cd ~/$DIR_YETI_CONTENT_BUNDLE/3dmark/bin/yeti
		
		echo Run the 3dmark application the way you would for Linux XCB:
		./3dmark --asset_root=../../assets -i ../../configs/gt1.json
	elif [[ $game -eq $GAME_DOOM ]] || [[ $game -eq $GAME_TR2 ]] ; then
		echo Following games: Doom/TR2 does not support non-stream test option.
		
	else
		echo "Invalid game: $game" 
		exit 1
	fi
elif [[ $option -eq $OPTION_STREAM_1PC ]] ; then
	echo "OPTION: STREAM 1PC ..." ; sleep $SLEEP_TIME
	if [[ $game -eq $GAME_3DMARK ]] ; then
		echo "GAME: 3DMARK ..." ; sleep $SLEEP_TIME
		if [[ $p4 == "t1"  ]] ; then
			echo "Terminal1." ; sleep $LEEP_TIME
			sudo uwf disable
			setPathLdLibraryPath
			setVkLoaderDisableYetiExtWhitelist
			echo Setup the swapchain for render+encode+stream:
			source ~/$DIR_ENG_BUNDLE_TO_USE/env/vce.sh
			cd ~/$DIR_YETI_CONTENT_BUNDLE/3dmark/bin/yeti
			
			echo "Type, but do not execute the following command:"
			echo "./3dmark --asset_root=../../assets -i ../../configs/gt1.json --output <output_full_path>"
			
			#NOTE: you can run a Yeti application with some debug output from the Vulkan loader and layers. To
			#do so, add VK_LOADER_DEBUG=all ahead of the application name. For example, for the 3dmark
			#command above, use:
			#VK_LOADER_DEBUG=all ./3dmark --asset_root=../../assets -i ../../configs/gt1.json
		elif [[ $p4 == "t2" ]] ; then
			echo "Terminal2". ; sleep $SLEEP_TIME
			clear
			echo setting up Yeti libraries...
			echo yeti 3dmark non-stream configuration run...
			pulseaudio --start

			setPathLdLibraryPath
			cd ~/$DIR_ENG_BUNDLE_TO_USE/bin	
			prompt_t2_with_ip $GAME_3DMARK $OPTION_LOCAL_IP

		elif [[ $p4 == "client" ]] ; then
			echo "Terminal3 / client"; sleep $SLEEP_TIME
			clear
			echo setting up Yeti on client machine...
			
			apt install -y libc++abi-dev

			setPathLdLibraryPath

			cd ~/$DIR_ENG_BUNDLE_TO_USE/bin
			echo "Type, but do not execute the following command:"
			echo "./game_client run-direct <IPv4 address of the Yeti computer>:44700"
		else
			echo "Invalid  p4 is slipped through: $p4."
			exit 1
		fi	
	elif [[ $game -eq $GAME_DOOM ]] ; then
		echo "GAME: DOOM..." ; sleep $SLEEP_TIME
		if [[ $p4 == "t1"  ]] ; then
			echo "Terminal1." ; sleep $SLEEP_TIME
			setPathLdLibraryPath
			setVkLoaderDisableYetiExtWhitelist
			source ~/$DIR_ENG_BUNDLE_TO_USE/env/vce.sh
			mkdir -p ~/doom/yeti-release
			cd ~/doom/yeti-release
			echo "Type, but do not execute the following command from this directory ~/doom/yeti-release:"
			echo "./DOOM"
		elif [[ $p4 == "t2" ]] ; then
			echo "Terminal2." ; sleep $SLEEP_TIME
			pulseaudio --start			
			setPathLdLibraryPath
			cd ~/$DIR_ENG_BUNDLE_TO_USE/bin
			prompt_t2_with_ip $GAME_DOOM $OPTION_LOCAL_IP
		elif [[ $p4 == "client" ]] ; then
			echo "Terminal3." ; sleep $SLEEP_TIME
			clear
			echo setting up Yeti on client machine...
			apt install -y libc++abi-dev
			setPathLdLibraryPath
			cd ~/$DIR_ENG_BUNDLE_TO_USE/bin
			echo "Type, but do not execute the following command:"
			echo "./game_client run-direct 127.0.0.1:44700"
			
		else
			echo "Unsupported terminal selection: $p4" ; exit 1
		fi 
	elif [[ $game -eq $GAME_TR2 ]] ; then
		echo "1PC streaming for TR2 is not added yet or will not be added. Please use 2PC streaming option."
	else
		echo "Unsupported game was specified: $game" ; exit 1
	fi
elif [[ $option -eq $OPTION_STREAM_2PC ]] ; then
	echo "OPTION: STREAM 2 PC." ; sleep $SLEEP_TIME

	if [[ $game -eq $GAME_3DMARK ]] ; then
		echo "GAME: 3DMARK." ; sleep $SLEEP_TIME
		if [[ $p4 == "t1" ]] ; then
			echo "Terminal1." ; sleep $SLEEP_TIME
			clear
			echo setting up Yeti libraries...
			echo yeti 3dmark non-stream configuration run...
			echo terminal 1...
			sleep $SLEEP_TIME
			sudo uwf disable

			setPathLdLibraryPath
			setVkLoaderDisableYetiExtWhitelist
			#setYetiDisableFabricatedConnected

			echo Setup the swapchain for render+encode+stream:
			source ~/$DIR_ENG_BUNDLE_TO_USE/env/vce.sh
			cd ~/$DIR_YETI_CONTENT_BUNDLE/3dmark/bin/yeti
			
			#NOTE: you can run a Yeti application with some debug output from the Vulkan loader and layers. To
			#do so, add VK_LOADER_DEBUG=all ahead of the application name. For example, for the 3dmark
			#command above, use:
			#VK_LOADER_DEBUG=all ./3dmark --asset_root=../../assets -i ../../configs/gt1.json
			
			echo Type, but do not execute the following command:
			echo "./3dmark --asset_root=../../assets -i ../../configs/gt1.json --output <output_full_path>"
		elif [[ $p4 == "t2" ]] ; then
			echo "Terminal2." ; sleep $SLEEP_TIME
			clear

			echo setting up Yeti libraries...
			echo yeti 3dmark non-stream configuration run...
			echo terminal 2...
			sleep $SLEEP_TIME
			pulseaudio --start

			setPathLdLibraryPath
			cd ~/$DIR_ENG_BUNDLE_TO_USE/bin
			displayIpv4
			prompt_t2_with_ip $GAME_3DMARK $OPTION_EXTERNAL_IP

		elif [[ $p4 == "client" ]] ; then
			echo "Terminal3 / client." ; sleep $SLEEP_TIME
			clear
			echo setting up Yeti on client machine...
			apt install -y libc++abi-dev

			setPathLdLibraryPath

			cd ~/$DIR_ENG_BUNDLE_TO_USE/bin
			echo "Type, but do not execute the following command:"
			echo "./game_client run-direct <IPv4 address of the Yeti computer>:44700"
		else
			echo "Invalid  p4 is slipped through: $p4."
			exit 1
		fi
	elif [[ $game -eq $GAME_TR2 ]] ; then
		if [[ $p4 == "t1" ]] ; then			
			echo "Terminal1." ; sleep $SLEEP_TIME
			rm -rf /usr/local/cloudcast/*
			rm -rf  ~/.local/share/vulkan/icd.d/*

			if [[  -z /etc/vulkan/icd.d/amd_icd64.json ]] ; then
				echo "Error: can not find /etc/vulkan/icd.d/amd_icd64.json..."
			fi

			sudo mkdir -p /usr/local/cloudcast
			sudo chown -R $(id -u):$(id -g) /usr/local/cloudcast
			sudo mkdir -p /var/game
			sudo chown -R $(id -u):$(id -g) /var/game
			sudo mkdir -p /srv/game
			sudo chown -R $(id -u):$(id -g) /srv/game

			# This static path will not work well!!!			
			# ln -s /cst_v320_test/drop-March-21-debian/test-apps/yeti/ggp-eng-bundle	 /usr/local/cloudcast

			sshpass -p amd1234 scp -v -o StrictHostKeyChecking=no -r root@$REPO_SERVER_IP:/$REPO_SERVER_LOCATION/ggp-eng-bundle/* /usr/local/cloudcast/
			#tar -xf /tmp/ggp-eng-bundle-20190413.tar.gz -C /usr/local/cloudcast --strip-components=1

			if [[ $? -ne 0 ]] ; then
				echo "Failed to copy ggp-eng-bundle"
				exit 1
			fi

			FILE_CLOUDCAST_COMMON=/usr/local/cloudcast/env/common.sh

			if [[ -z $FILE_CLOUDCAST_COMMON ]] ; then
				echo "Error: Can not find $FILE_CLOUDCAST_COMMON"
				exit 1
			else
				echo "Adding export variable VK_ICD_FILESNAMES to $FILE_CLOUDCAST_COMMON"

				# Alan mentioned this line is wrong on e-mail  5/20/2019 and use the line below, however it was running 
				# ok with Sam
				#echo "export VK_ICD_FILENAMES=/etc/vulkan/icd.d/amd_icd64.json" >>  /usr/local/cloudcast/env/common.sh

				# 5.20.2019 Alan, this turned out to be wrong.
				# Replaced with following.

				#echo "export VK_ICD_FILENAMES=/etc/vulkan/icd.d/amd_icd64.json" >>  /usr/local/cloudcast/env/common.sh
				echo "export GGP_INTERNAL_VK_ICD_DELEGATE=/opt/amdgpu-pro/lib/x86_64-linux-gnu/amdvlk64.so" >>  /usr/local/cloudcast/env/common.sh
			fi	

			if [[ ! -d ~/tr2 ]] ; then
				echo "~/tr2 does not exist."
				mkdir -p ~/tr2
				echo "Copying tr2 from $REPO_SERVER_IP, will take some time..."
				sshpass -p amd1234 scp -v -r -o StrictHostKeyChecking=no root@$REPO_SERVER_IP:$REPO_SERVER_LOCATION/tr2/* ~/tr2/

				if [[ $? -ne 0 ]] ; then
					echo "Failed to copy tr2..."
					exit 1
				fi
			else
				echo "~/tr2 exists, skipping."
			fi

			sleep 5 

			ln -s ~/tr2 /srv/game/assets
			cd /usr/local/cloudcast	
			
			# Slightly possible tweaked out content of ./runit/catchingfire.sh is below.
			# No longer see in the instruction.

			#./runit/catchingfire.sh

			source ./env/vce.sh

			if [[ ! -d /var/game ]]; then
				echo "Create directory /var/game."
  				exit 1
			fi
			
			cd /srv/game/assets/

			if [[ ! -f /srv/game/assets/TR2_yeti_final ]]; then
  				echo "Unpack the catching fire package to /srv/game/assets/ (or symlink)"
	  			exit 1
			fi
			
			echo "type the following to run the catching fire."
			pushd /srv/game/assets/
			./TR2_yeti_final
			popd

		elif [[ $p4 == "t2" ]] ; then
			echo "Terminal2." ; sleep $SLEEP_TIME
                        displayIpv4
                        prompt_t2_with_ip $GAME_TR2 $OPTION_EXTERNAL_IP
			cd /usr/local/cloudcast
		elif [[ $p4 == "client" ]] ; then
			echo "game client from Linux is dropped support. Please use windows version."
			exit 0
		else 
			echo "Invalid terminal selected: $p4 " ; exit 1
		fi

	elif [[ $game -eq $GAME_DOOM ]] ; then
		echo "GAME: DOOM" ; sleep $SLEEP_TIME
		if [[ $p4 == "t1" ]] ; then			
			echo "Terminal1." ; sleep $SLEEP_TIME

			if [[ ! "$(ls -A ~/$DIR_ENG_BUNDLE_TO_USE)" ]] ; then
    			echo "<path> is empty!"
			else
    			echo "<path> is not empty"
			fi

			setPathLdLibraryPath
			setYetiDisableFabricatedConnected

			source ~/$DIR_ENG_BUNDLE_TO_USE/env/vce.sh
			mkdir -p ~/doom/yeti-release

                        if [[ ! -f ~/doom/yeti-release/DOOM ]] ; then
				mkdir -p ~/doom/yeti-release/
                                echo "the DOOM is not in ~/doom/yeti-release, copying, will take some time..."
				sshpass -p amd1234 scp -v -o StrictHostKeyChecking=no -r root@$REPO_SERVER_IP:/$REPO_SERVER_LOCATION/Doom_Linux/* ~/doom/yeti-release/

				if [[ $? -ne 0 ]] ; then
					echo "Failed to copy DOOM"
					exit 1
				fi

                        fi

			cd ~/doom/yeti-release
			echo "Type, but do not execute the following command"
			echo "./DOOM"
		elif [[ $p4 == "t2" ]] ; then
			echo "Terminal2." ; sleep $SLEEP_TIME
			pulseaudio --start

			if [[ $? != 0 ]] ; then 
				echo "Failed to run pulseaudio, does it exist or some other problem? return code: $?"
				exit 1
			fi
			
			setPathLdLibraryPath

			cd ~/$DIR_ENG_BUNDLE_TO_USE/bin

			if [[ $? != 0 ]] ; then 
				echo "Failed to cd into ~/$DIR_ENG_BUNDLE_TO_USE, does it exist? return code: $?"
				exit 1
			fi

			displayIpv4
			prompt_t2_with_ip $GAME_DOOM $OPTION_EXTERNAL_IP

		elif [[ $p4 == "client" ]] ; then
			echo "Terminal3 / client." ; sleep $SLEEP_TIME
			echo "game client from Linux is dropped support. Please use windows version."
			exit 0

			export LD_LIBRARY_PATH=~/$DIR_ENG_BUNDLE_TO_USE/lib

			setPathLdLibraryPath

			cd ~/$DIR_ENG_BUNDLE_TO_USE/bin

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

