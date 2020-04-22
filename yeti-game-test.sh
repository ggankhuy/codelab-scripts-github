#    Launcher scripts for 3DMARK, DOOM and TR2.
#    To run the script simply provide root location of drop package in ~./bashrc
#    This script has been tested with root account only and assumes everything is copied to root folder.
#    Non-root account has not been tested.
#    i.e. in ~/.bashrc:
#    export GIB_DROP_ROOT=/work/drop-March-21-debian/

#    After that ./yeti-game-setup.sh setup to run preliminary setup for either 3dmark or DOOM.
#    After that from each of terminal 1,2 and 3 (client) 
#    . ./yeti-game-setup doom yeti 2 t1
#    . ./yeti-game-setup doom yeti 2 t2
#    . ./yeti-game-setup doom yeti 2 t1t2
#    At the end of each run, the script will prompt with the last syntax to run the actual game for each terminals
#    with seconds apart for easier copy and paste and launch (type but not run).

#    print bar.
#    Prints usage information.

source ./common.sh

#    Actual scripts starts here.

p1=$1       # game name    
p2=$2       # yeti or linux.
p3=$3       # 0 - non-stream, 1 for 1 pc stream , 2 for 2 pc stream.
p4=$4       # t1 - for terminal 1 (obsolete), t2 for terminal 2(obsolete), t1t2 for both terminal in one terminal.
            # $4 is not applicable if $3 is non streaming.

#   qts servers: ens3
#   daytona x2: ens7
#   gb 02: ens8

CONFIG_EXT_INT=ens7

for var in "$@"
do
    if [[ ! -z `echo "$var" | grep "int="` ]]  ; then
        echo "external interface: $var"
        CONFIG_EXT_INT=`echo $var | cut -d '=' -f2`
        echo "CONFIG_EXT_INT: $CONFIG_EXT_INT"
    fi
done

sleep 3

game=0            # game
mode=0            # 0 for yeti, 1 for linux
option=0    # 0 for streaming, 1 and 2 for streaming with 1 or 2 pc respectively.
n=0

MODE_YETI=0
MODE_LINUX=1

OPTION_NOSTREAM=0
OPTION_STREAM_2PC=2

TERMINAL_T1=0
TERMINAL_T2=1
TERMINAL_CLIENT=2

SLEEP_TIME=1

#    Used for remote initialization of game environment in addition to setup.

CONFIG_ABORT_GAME=1

CONFIG_ITERATION_3DMARK=10
CONFIG_ITERATION_CONGA=1
CONFIG_POLICY_DIR=/usr/local/cloudcast/dev/bin/
vm_check
sleep $SLEEP_TIME

#    apt packages 

sudo apt install sshpass dstat net-tools -y

#    Process help request. 

if [[ $1 == "--help"  ]] || [[ -z $1 ]] ; then
    usage
    exit 1
fi

#    p1 is for either 3dmark or doom (unless more support added.

if [[ $p1 == "3dmark" ]] ; then
    echo "3dmark is selected..."
    game=$GAME_3DMARK
elif  [[ $p1 == "doom" ]] ; then
    echo "doom is selected..."
    game=$GAME_DOOM
elif [[ $p1 == "tr2" ]] ; then
    echo "tr2 is selected..."
    game=$GAME_TR2
elif [[ $p1 == "quail" ]] ; then
    echo "tr2 is selected..."
    game=$GAME_QUAIL
elif [[ $p1 == "conga" ]] ; then
    echo "conga is selected..."
    game=$GAME_CONGA
elif [[ $p1 == "setup" ]] ; then
    echo "setting up the system for test."
    echo "p2: $p2..."
    common_setup $p2
    exit 0
else
    echo "Invalid game selected: $p1"
    exit 1
fi

if [[ -z $2  ]] || [[ -z $3 ]] ; then
    echo "p0 selected is for game not setup. Therefore you need to supply p2-4 parameters."
    usage
    exit 1

    if [[ -z $4 ]] && [[ $3 -ne "0" ]] ; then
        echo "if p3 is not zero, then p4 must be selected."
        exit 1
    fi
fi

#    p2 is for either linux or yeti.

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

#    p3 is for  no-stream, stream 1 pc or 2 pc option.
#    With no stream         p3=0, terminal is not needed to be specified in p4.
#    With stream 1 pc    p3=1, terminal need to be specified with either p4=t1, t2 or client.
#    With stream 2 pc    p3=2, terminal need to be specified with either p4=t1, t2 or client.

if [[ $p3 -eq 2 ]] && [[ -z $p4 ]] ; then
    echo "You selected <option>=2 but then left <terminal type> empty. Exiting.."
    exit 1
fi

if [[ $p3 -eq  0 ]] ; then
    echo "no stream option is selected."
    option=$OPTION_NO_STREAM
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

#    In bashrc, cd into /git.co/ad-hoc-scripts/

ret=`cat ~/.bashrc  | grep "cd /git.co/ad-hoc-scripts"`

if [[ -z $ret  ]] ; then
        echo "Inserting cd /git.co/ad-hoc-scripts to bash..."
        echo "cd /git.co/ad-hoc-scripts" >> ~/.bashrc
fi

#    Load amdgpu, kfd driver:

#         sudo modprobe amdkfd

sudo modprobe amdgpu
ret2=`lsmod | grep -u ^amdgpu`

if [[ -z $ret2 ]] ; then
        echo "Failed to install amdgpu (modprobe amdgpu), check the driver is installable or GPU is present."
        exit 1
else
        echo lsmod amdgpu: $ret2
fi

GAME_PARAM="-"

if [[ $game -eq $GAME_3DMARK ]] ; then
    echo "GAME: 3DMARK." ; sleep $SLEEP_TIME
    SOURCE_FOLDER=3dmark
    DESTINATION_FOLDER=./cumbia
    GAME_EXECUTABLE=3dmark
    GAME_FOLDER=./
    GAME_NAME=$GAME_3DMARK
    GAME_PARAM="--asset_root=../../assets -i ../../configs/gt1.json"
elif [[ $game -eq $GAME_QUAIL ]] ; then
    echo "TR2 is selected" ; sleep $SLEEP_TIME
    SOURCE_FOLDER=Quail
    DESTINATION_FOLDER=infiltrator
    GAME_EXECUTABLE=InfiltratorDemo.elf
    GAME_FOLDER=./InfiltratorDemo/Binaries/Quail/
    GAME_NAME=$GAME_QUAIL
elif [[ $game -eq $GAME_TR2 ]] ; then
    echo "TR2 is selected" ; sleep $SLEEP_TIME
    SOURCE_FOLDER=tr2
    DESTINATION_FOLDER=catchingfire
    GAME_EXECUTABLE=TR2_yeti_final
    GAME_FOLDER="./"
    GAME_NAME=$GAME_TR2
elif [[ $game -eq $GAME_DOOM ]] ; then
    echo "GAME: DOOM" ; sleep $SLEEP_TIME
    SOURCE_FOLDER=Doom_Linux
    DESTINATION_FOLDER=lincoln
    GAME_EXECUTABLE=DOOM
    GAME_FOLDER="./"
    GAME_NAME=$GAME_DOOM
elif [[ $game -eq $GAME_CONGA ]] ; then
    echo "GAME: CONGA" ; sleep $SLEEP_TIME
    SOURCE_FOLDER=Conga
    DESTINATION_FOLDER=conga
    GAME_EXECUTABLE=benchmark
    GAME_FOLDER="./"
    GAME_NAME=$GAME_CONGA

else
    echo "Unsupported game: $game" ; exit 1
fi

if [[ $game -eq $GAME_3DMARK ]] ; then

    echo "GAME: 3DMARK." ; sleep $SLEEP_TIME
    SOURCE_FOLDER=3dmark
    DESTINATION_FOLDER=./cumbia
    GAME_EXECUTABLE=3dmark
    GAME_FOLDER=./
    GAME_NAME=$GAME_3DMARK
    #GAME_PARAM="--asset_root=../../assets -i ../../configs/gt1.json --output <output_full_path>"
    sudo mkdir -p /log/3dmark/
        sudo chmod -R g=u /log/3dmark
        sudo chmod -R o=u /log/3dmark

    GAME_PARAM="--asset_root=../../assets -i ../../configs/gt2.json --output /log/3dmark/3dmark.$DATE.log"

elif [[ $game -eq $GAME_CONGA ]] ; then
    echo "GAME: CONGA." ; sleep $SLEEP_TIME
    sudo mkdir -p /log/conga/
        sudo chmod -R g=u /log/conga
        sudo chmod -R o=u /log/conga
    #GAME_PARAM="--asset_root=/srv/game/assets/conga -i /srv/game/assets/conga/example_settings/demo_loop.json --output /log/conga/conga.$DATE.log" 
    GAME_PARAM="--asset_root=/srv/game/assets/ -i /srv/game/assets/example_settings/demo_loop.json --output /log/conga/conga.$DATE.log" 

elif [[ $game -eq $GAME_DOOM ]] || [[ $game -eq $GAME_TR2 ]] ; then
    echo Following games: Doom/TR2 does not support non-stream test option.
    
else
    echo "Invalid game: $game" 
    exit 1
fi

if [[ $game -eq $GAME_QUAIL ]] ; then
    sudo rm /srv/game/assets/
    sudo mkdir -p /srv/game/assets/
    sudo ln -fs /srv/game/$DESTINATION_FOLDER/ /srv/game/assets/Quail
else
    sudo rm /srv/game/assets
    sudo mkdir -p /srv/game
    sudo ln -fs /srv/game/$DESTINATION_FOLDER /srv/game/assets
fi

copy_game_files $SOURCE_FOLDER /srv/game/$DESTINATION_FOLDER/

# infiltrator specific code.

if [[ $game -eq $GAME_QUAIL ]] ; then
    echo "Quail specific steps..."
    sudo mkdir -p /srv/game/assets/InfiltratorDemo/Content/Paks
    sudo ln -fs /srv/game/assets/Quail/InfiltratorDemo/Content/Paks/InfiltratorDemo-Quail.pak \
    /srv/game/assets/InfiltratorDemo/Content/Paks/InfiltratorDemo-Quail.pak
    sudo chmod a+x /srv/game/assets/Quail/InfiltratorDemo/Binaries/Quail/*
elif [[ $game -eq $GAME_DOOM ]] ; then
    echo "DOOM specific steps..."
    sudo chmod 755 /srv/game/$DESTINATION_FOLDER/DOOM
fi

cd /usr/local/cloudcast    

if [[ ! -d /var/game ]]; then
    echo "Create directory /var/game."
    exit 1
fi

sudo chmod -R g=u /usr/local/cloudcast/
sudo chmod -R o=u /usr/local/cloudcast/
sudo chmod -R g=u /srv/game/
sudo chmod -R o=u /srv/game/

cd /srv/game/assets/

if [[ $game -eq $GAME_3DMARK ]] ; then
    echo "3dmark specific steps..."
    cd /srv/game/assets/bin/yeti
elif [[ $game -eq $GAME_QUAIL ]] ; then
    cd /srv/game/assets/Quail
fi

if [[ $option -eq $OPTION_NOSTREAM ]] ; then
    echo "OPTION: NON-STREAM." ; sleep $SLEEP_TIME

    sudo chmod 755 ./$GAME_FOLDER/$GAME_EXECUTABLE    
    echo chmod 755 ./$GAME_FOLDER/$GAME_EXECUTABLE

    if [[ $game -eq $GAME_3DMARK ]] ; then
        echo "3dmark specific steps..."
        common_runtime_setup novce

        for (( n=0; n < $CONFIG_ITERATION_3DMARK; n++ )) ; do
            echo Running 3dmark for $n th time.
            DATE_3DMARK_LOOP=`date +%Y%m%d-%H-%M-%S`
            sleep 3
    
                sudo sed -i '/encode_width/c \ \encode_width: 1920' $CONFIG_POLICY_DIR/lan_policy.proto_ascii
                sudo sed -i '/encode_height/c \ \encode_height: 1080' $CONFIG_POLICY_DIR/lan_policy.proto_ascii
                sudo sed -i '/resolution/c \ \"resolution" : "1920x1080",' ../../configs/gt1.json
                sudo sed -i '/resolution/c \ \"resolution" : "1920x1080",' ../../configs/gt2.json 
            ./3dmark --asset_root=../../assets -i ../../configs/gt1.json  --output /log/3dmark/3dmark.1080p.gt1.$DATE_3DMARK_LOOP.log
            ./3dmark --asset_root=../../assets -i ../../configs/gt2.json  --output /log/3dmark/3dmark.1080p.gt2.$DATE_3DMARK_LOOP.log
                sudo sed -i '/encode_width/c \ \encode_width: 1280' $CONFIG_POLICY_DIR/lan_policy.proto_ascii
                sudo sed -i '/encode_height/c \ \encode_height: 720' $CONFIG_POLICY_DIR/lan_policy.proto_ascii
                sudo sed -i '/resolution/c \ \"resolution" : "1280x720",' ../../configs/gt1.json
                sudo sed -i '/resolution/c \ \"resolution" : "1208x720",' ../../configs/gt2.json 
            ./3dmark --asset_root=../../assets -i ../../configs/gt1.json  --output /log/3dmark/3dmark.720p.gt1.$DATE_3DMARK_LOOP.log
            ./3dmark --asset_root=../../assets -i ../../configs/gt2.json  --output /log/3dmark/3dmark.720p.gt2.$DATE_3DMARK_LOOP.log
                sudo sed -i '/encode_width/c \ \encode_width: 3840' $CONFIG_POLICY_DIR/lan_policy.proto_ascii
                sudo sed -i '/encode_height/c \ \encode_height: 2160' $CONFIG_POLICY_DIR/lan_policy.proto_ascii
                sudo sed -i '/resolution/c \ \"resolution" : "3840x2160",' ../../configs/gt1.json
                sudo sed -i '/resolution/c \ \"resolution" : "3840x2160",' ../../configs/gt2.json 
            ./3dmark --asset_root=../../assets -i ../../configs/gt1.json  --output /log/3dmark/3dmark.4k.gt1.$DATE_3DMARK_LOOP.log
            ./3dmark --asset_root=../../assets -i ../../configs/gt2.json  --output /log/3dmark/3dmark.4k.gt2.$DATE_3DMARK_LOOP.log
        done
    elif [[ $game -eq $GAME_CONGA ]] ; then
        echo "conga specific steps..."
        common_runtime_setup novce
        for (( n=0; n < $CONFIG_ITERATION_CONGA; n++ )) ; do
            #GAME_PARAM="--asset_root=/srv/game/assets/conga -i /srv/game/assets/conga/example_settings/demo_loop.json --output /log/conga/conga.$DATE.log" 
            GAME_PARAM="--asset_root=/srv/game/conga -i /srv/game/conga/example_settings/demo_loop.json --output /log/conga/conga.$DATE.log" 
            #./benchmark --asset_root=/srv/game/assets -i /srv/game/assets/example_settings/demo_loop.json
            ./benchmark --asset_root=/srv/game/assets -i /srv/game/assets/example_settings/demo_loop.json
        done
    else
        echo "Unsupport nonstreaming game: $game"
    fi
    
elif [[ $option -eq $OPTION_STREAM_2PC ]] ; then
    echo "OPTION: STREAM 2 PC." ; sleep $SLEEP_TIME

    common_runtime_setup vce
    
    if [[ $p4 == "t1" ]] || [[ $p4 == "t1t2" ]] || [[ $p4 == "nolaunch" ]] ; then            
        echo "Terminal1." ; sleep $SLEEP_TIME
    
            copy_game_files $SOURCE_FOLDER /srv/game/$DESTINATION_FOLDER/

        if [[ $game -eq $GAME_3DMARK ]] ; then
            echo "3dmark specific steps..."
            cd /srv/game/assets/bin/yeti
        elif [[ $game -eq $GAME_QUAIL ]] ; then
            cd /srv/game/assets/Quail
        fi
        if [[ $p4 == "nolaunch" ]] ; then
            echo "No launch..."
        fi 

        if [[ $p4 == "t1t2" ]] ; then
            process_t1t2 $GAME_EXECUTABLE $GAME_FOLDER "$GAME_PARAM" "$CONFIG_EXT_INT"
        else
            echo ./$GAME_EXECUTABLE
        fi
    elif [[ $p4 == "t2" ]] ; then
        echo "Terminal2." ; sleep $SLEEP_TIME
            displayIpv4
            prompt_t2_with_ip $GAME_NAME $OPTION_EXTERNAL_IP $CONFIG_EXT_INT
        cd /usr/local/cloudcast
    elif [[ $p4 == "client" ]] ; then
        echo "game client from Linux is dropped support. Please use windows version."
        exit 0
    else 
        echo "1. Invalid terminal selected: $p4 " ; exit 1
    fi
else
    echo "Invalid option is slipped through."
    exit 1
fi

