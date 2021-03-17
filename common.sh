# 	game defines.

SLEEP_TIME=1
SLEEP_TIME_2=2
GAME_3DMARK=0
GAME_DOOM=1
GAME_TR2=2
GAME_QUAIL=3
GAME_CONGA=4
GAME_ODIN=5
GAME_CHASE=6
GAME_MIYAGI=7
GAME_GEORGETOWN=8
GAME_HELLO_GGP_STANDALONE=9
GAME_VK_EXAMPLE=10
DATE=`date +%Y%m%d-%H-%M-%S`

OPTION_EXTERNAL_IP=1
OPTION_LOCAL_IP=2
REPO_SERVER_IP=""

#   When set, could be useful when repo server is not reachable and no need as setup is already done.'

CONFIG_BYPASS_SETUP_REPO_SERVER=0

CONFIG_POLICY_DIR=/usr/local/cloudcast/dev/bin/

# 	sup resolution defines.

RESOLUTION_1080=1080
RESOLUTION_720=720
RESOLUTION_4K=4k
RESOLUTIONS_SUPPORTED=( $RESOLUTION_720 $RESOLUTION_1080 $RESOLUTION_4K )
STREAMER_POLICY_FILE=lan_policy.proto_ascii

# 0 - for tar
# 1 - for deb
# 2 - no copy or invalid choice.

OPTION_GGP_INSTALL_USE_DEB=1

#     game repo servers.

REPO_SERVER_IPS=("11.0.0.30" "10.216.66.54" "10.216.66.51" "10.217.75.124" "10.216.54.38" "10.217.73.160")
REPO_SERVER_LOCATION=/repo/stadia

OPTION_DHCLIENT_EXT_INT=1

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

#	file cfines.

FILE_COPY_SCP=1
FILE_COPY_WGET=2
FILE_COPY_RSYNC=3
FILE_COPY_NONE=100
OPTION_FILE_COPY_PROTOCOL=$FILE_COPY_RSYNC

export DIR_YETI_CONTENT_BUNDLE=yeti-content-bundle
export DIR_GGP_ENG_BUNDLE=ggp-eng-bundle
export GGP_BUNDLE_VERSION=ggp-eng-bundle-20190413.tar.gz

if [[ $OPTION_GGP_INSTALL_USE_DEB -eq 1 ]] ; then
    export GGP_BUNDLE_VERSION=ggp-eng-bundle-20190829.deb
#    export GGP_BUNDLE_VERSION=ggp-eng-bundle-20200325.deb
#    export GGP_BUNDLE_VERSION=ggp-eng-bundle_20200910.1.0.deb
elif [[ $OPTION_GGP_INSTALL_USE_DEB -eq 0 ]] ; then
    export GGP_BUNDLE_VERSION=ggp-eng-bundle-20190518.tar.gz
else
    echo "Invalid value for OPTION_GGP_INSTALL_USE_DEB, only 0 or 1 allowed: $OPTION_GGP_INSTALL_USE_DEB"
    echo "Leaving the value of GGP_BUNDLE_VERSION unchanged as: $GGP_BUNDLE_VERSION"
fi

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
#    $1 - full path of file or directory to copy
#    $2 - destination  on local  file system to copy to.
    
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

#    Check if the calling script is running inside VM
#    input:     None
#    return  exit 1 if not running on VM.

function vm_check () {
#    Initialization function used by yeti-game-test.sh. 
#    input: $1 - GIB_DROP_ROOT location.
#    return: 1 - on any error.
#    r    0 - on success.    

    # Check if running in VM, if not, exit with error.

    sudo apt install virt-what -y 

    if [[ -z `sudo which virt-what` ]] ; then
        echo "Failed to install virt-what..."
        exit 1
    fi

    if [[ -z `sudo virt-what` ]] ; then
        echo "virt-what returns null, it is not running inside vm: hostname: "
        echo `hostname`
        exit 1
    else
        echo "OK, running inside VM..."
    fi

    sleep $SLEEP_TIME
}

function common_runtime_setup ()
{
    echo commont_runtime_setup

    if [[ $1 == "vce" ]] ; then
        echo "setting vce..."
        source /usr/local/cloudcast/env/vce.sh
    elif [[ $1 == "novce" ]] ; then
        echo "setting non vce..."
        if [[ -z `echo /usr/local/cloudcast/env/vce_nostreamer.sh | grep YETI_FORCE_SWAPCHAIN | grep null` ]] ; then
                    echo "export YETI_FORCE_SWAPCHAIN=\"null\"" >> /usr/local/cloudcast/env/vce_nostreamer.sh
        fi
        source /usr/local/cloudcast/env/vce_nostreamer.sh
    else
        echo "common_runtime_setup: invalid p1: $1, supported values are vce and novce."
        exit 1
    fi

    export GGP_INTERNAL_VK_DELEGATE_ICD=/opt/amdgpu-pro/lib/x86_64-linux-gnu/amdvlk64.so
    export GGP_INTERNAL_VK_ALLOW_GOOGLE_YETI_SURFACE=1
    export GGP_INTERNAL_ENABLE_FABRICATED_PLAYER_MESSAGES=1 
    export AMD_VK_USE_PIPELINE_CACHE=true 
    export XDG_CACHE_HOME="/mnt/developer" 
    export GGP_INTERNAL_VK_DISABLE_VSYNC=1
    export GGP_INTERNAL_VK_FORCE_PRESENT_MODE=1
    export AMDVLKXF=$AMDVLKXF,YUV-9c240ad2 # only for 24 or older vulkan driver when using 20200910.deb. 

    export GGP_VK_FORCE_PRIVATE_ASYNC_COMPUTE_SWAPCHAIN_PRESENTS=1
    export GGP_VK_DISABLE_UNIVERSAL_QUEUE_PRESENTS=1
    export YETI_VK_FORCE_PRIVATE_ASYNC_COMPUTE_SWAPCHAIN_PRESENTS=1
    export YETI_VK_DISABLE_UNIVERSAL_QUEUE_PRESENTS=1
    sleep 1
}

function set_repo_server() {
    echo "Determining reachable repo server..."

    # Check if it is already setup in ~/.bashrc

    REPO_SERVER_IP_BASHRC=`cat ~/.bashrc | grep REPO_SERVER_IP`

    if [[ -z $REPO_SERVER_IP_BASHRC ]] ; then
        echo "REPO_SERVER_IP is not setup in bashrc."

    REPO_SERVER_IP=""

        for (( i=0 ; i < ${#REPO_SERVER_IPS[@]} ; i++ ))
        do
            ping -c 4 ${REPO_SERVER_IPS[$i]}
            stat=$?

            if [[ $stat -eq 0 ]] ; then
                    echo "Found reachable repo server: ${REPO_SERVER_IPS[$i]}"
                    REPO_SERVER_IP=${REPO_SERVER_IPS[$i]}
                    break
            fi
        done
    
    if [[ -z $REPO_SERVER_IP ]] ; then
        echo "Error: can not find pingable repo server IP:"
    else
            echo "repo server is set to: $REPO_SERVER_IP"
            echo "REPO_SERVER_IP=$REPO_SERVER_IP" >> ~/.bashrc
    fi
    else
        echo "REPO_SERVER_IP is already setup in bashrc: $REPO_SERVER_IP_BASHRC"
        REPO_SERVER_IP=`echo $REPO_SERVER_IP_BASHRC | cut -d '=' -f2`
        echo "REPO_SERVER_IP from bashrc is set to: $REPO_SERVER_IP"
    fi
}

function common_setup () {
    clear
    echo "Setup Yeti system for 3dmark on ubuntu 1604 / 1803..."

    sudo apt install sshpass

    if [[ $? -ne 0 ]] ; then
        echo "Error: Failed to install sshpass."
        exit 1
    fi

    sleep $SLEEP_TIME

    echo "Copying ggp-eng-bundle to /usr/local/cloudcast..."
    

    if [[ $CONFIG_BYPASS_SETUP_REPO_SERVER -eq 0 ]] ; then
        set_repo_server 
    fi

    if [[ $OPTION_FILE_COPY_PROTOCOL == $FILE_COPY_RSYNC ]] ; then
       sudo sshpass -p amd1234 rsync -v -z -r -e "ssh -o StrictHostKeyChecking=no" root@$REPO_SERVER_IP:/$REPO_SERVER_LOCATION/florida/$GGP_BUNDLE_VERSION /tmp/
    elif [[ $OPTION_FILE_COPY_PROTOCOL == $FILE_COPY_SCP ]] ; then
       sudo sshpass -p amd1234 scp -C -v -o StrictHostKeyChecking=no -r root@$REPO_SERVER_IP:/$REPO_SERVER_LOCATION/florida/$GGP_BUNDLE_VERSION /tmp/
    else
       echo "ERROR: Unknown or unsupported copy protocol."
    fi
    
    if [[ $? -ne 0 ]] ; then
       echo "Failed to rsync copy ggp-eng-bundle"
       exit 1
    fi

    sudo mkdir -p /usr/local/cloudcast
    sudo chown -R $(id -u):$(id -g) /usr/local/cloudcast
    sudo mkdir -p /var/game
    sudo chown -R $(id -u):$(id -g) /var/game
    sudo mkdir -p /srv/game
    sudo chown -R $(id -u):$(id -g) /srv/game

    if [[ $OPTION_GGP_INSTALL_USE_DEB -eq 1 ]] ; then
        echo "ggp bundle is installed through debian package..."
        sudo dpkg -i /tmp/$GGP_BUNDLE_VERSION
        sleep 3
    elif [[ $OPTION_GGP_INSTALL_USE_DEB -eq 0 ]] ; then
        tar -xf /tmp/$GGP_BUNDLE_VERSION -C /usr/local/cloudcast --strip-components=1
    else
        echo "Specified option 2 for OPTION_GGP_INSTALL_USE_DEB or invalid. Proceeding without copying."
    fi

    sudo mkdir /log
    sudo chmod a+rw /log
    
    apt-get install freeglut3 pulseaudio libpulse-dev
    
    echo "Soft links: "
    ls -l /usr/local/cloudcast/
    ls -l /opt/amdgpu-pro/lib/x86_64-linux-gnu/amdvlk64.so

    # If logic is not working...After add, it adds again. 

    if [[ -z `cat ~/.bashrc | grep "cd.*ad-hoc-scrits"` ]] ; then
        echo "adding to bashrc: cd `pwd`"
        sudo echo "cd `pwd`" >> ~/.bashrc
    else
        sudo echo "already in bashrc: cd `pwd`"
    fi

    sudo usermod -aG video $LOGNAME
    echo "video group: "
    echo `sudo getent group video`
    sleep $SLEEP_TIME
}

function prompt_t2_with_ip () {
    echo "Type, but do not execute the following command:"

    CONFIG_EXT_INT=$3
    if [[ $OPTION_DHCLIENT_EXT_INT -eq 1 ]] ; then
        sudo dhclient $CONFIG_EXT_INT
    else
        sudo ifup $CONFIG_EXT_INT
    fi

    if [[ $? -ne 0 ]] ; then
            echo "Warning: dhclient or ifup $CONFIG_EXT_INT failed. $CONFIG_EXT_INT interface might not have been able to get DHCP IP..."
    fi
    
    echo "ext interface: $CONFIG_EXT_INT"
    sleep 3
    external_ip=`ifconfig $CONFIG_EXT_INT | grep "inet " | tr -s " " | cut -d ' ' -f3`
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

        echo "./dev/bin/yeti_streamer -policy_config_file dev/bin/$STREAMER_POLICY_FILE -connect_to_game_on_start -direct_webrtc_ws -external_ip=$IP_TO_DISPLAY -port 44700 -null_audio=true"
}

function t1()
{
    echo "t1..."
}
#    Function used to process both terminal 1 (game itself) and terminal 2 (streaming server) from same shell window.
#    input:     $1 - name of the game executable.
#        $2 -parameter following game executable.
#    return: 1 - on any error.

function process_t1t2 ()
{
    ENABLE_LOG=0
    GAME=$1
    GAME_FOLDER=$2
    GAME_PARAM=$3
    GAME_RESO=$5

    echo "GAME: $GAME" 
    echo "GAME Params: $GAME_PARAM"
    echo "GAME folder: $GAME_FOLDER" 
    echo "CONFIG_EXT_INT: $CONFIG_EXT_INT"
    echo "GAME_RESO: $GAME_RESO"

    echo "external interface: $CONFIG_EXT_INT"
    sleep 1
    external_ip=`sudo ifconfig $CONFIG_EXT_INT | grep "inet " | tr -s " " | cut -d ' ' -f3`
    echo "external IP: " $external_ip

    set_resolution $GAME_RESO

    sleep 3

    DATE=`date +%Y%m%d-%H-%M-%S`
        LOG_DIR=/g/$DATE
        sudo mkdir -p $LOG_DIR
    sudo chmod 777 $LOG_DIR

    if [[ $OPTION_DHCLIENT_EXT_INT -eq 1 ]] ;  then
        echo "Configuring $CONFIG_EXT_INT..."
        sudo dhclient $CONFIG_EXT_INT
    else
        echo "Not configuring $CONFIG_EXT_INT..."
        sudo ifup $CONFIG_EXT_INT
    fi

    echo "./$GAME_FOLDER/$GAME $GAME_PARAM"

    read -p "Press a key to start $GAME..."

    sudo chmod 755 ./$GAME_FOLDER/$GAME

    if [[ $ENABLE_LOG -eq 0 ]] ;  then
            ./$GAME_FOLDER/$GAME $GAME_PARAM &
    else
            ./$GAME_FOLDER/$GAME $GAME_PARAM > $LOG_DIR/$GAME-$DATE.log &
    fi

    if [[ $? -ne 0 ]] ; then
            echo "Warning: dhclient $CONFIG_EXT_INT failed. $CONFIG_EXT_INT interface might not have been able to get DHCP IP..."
    fi

    if [[ -z $external_ip ]] ; then
            echo "Failed to get external IP: "  $external_ip
            exit 1
    fi

    sleep $SLEEP_TIME
    IP_TO_DISPLAY="$external_ip"
    cd /usr/local/cloudcast
    read -p "Press a key to start $GAME streaming server..."

    if [[ $ENABLE_LOG -eq 0 ]] ;  then
        ./dev/bin/yeti_streamer \
            -policy_config_file dev/bin/$STREAMER_POLICY_FILE \
            -connect_to_game_on_start -direct_webrtc_ws -external_ip=$IP_TO_DISPLAY \
            -port 44700 -null_audio=true 
    else
        ./dev/bin/yeti_streamer \
            -policy_config_file dev/bin/$STREAMER_POLICY_FILE \
            -connect_to_game_on_start -direct_webrtc_ws -external_ip=$IP_TO_DISPLAY \
            -port 44700 -null_audio=true > $LOG_DIR/$GAME-stream-$DATE.log
    fi
}

#	additional processing fork scores.
#	applico 3dmark only.

function display_result() {
    GAME=$1
    DEBUG_DISPLAY_RESULT=0

    echo "display_result: $GAME"    

    if [[ $GAME -eq $GAME_3DMARK ]] ; then 
        for i in gt1 gt2
        do
            for j in 720 1080 4k
            do
                echo ---------------- | tee -a /log/3dmark/$DATE.brief.log
                echo "$i:$j" | tee -a /log/3dmark/$DATE.brief.log

                egrep -irn "value\"" /log/3dmark/* | grep -i $i | grep -i $j | grep -v brief | tee -a /log/3dmark/$DATE.brief.log
                scores=`egrep -irn "value\"" /log/3dmark | grep -i $i | grep -i $j | grep -v brief | tr -s ' ' | cut -d ":" -f4`
                scores_count=`egrep -irn "value\"" /log/3dmark | grep $i | grep -i $j | grep -v brief | wc -l`

                if [[ $scores_count -eq 0 ]] ; then 
                    echo "unable to find scores for $i:$j" | tee -a /log/3dmark/$DATE.brief.log
                    continue
                fi
                scores_cumulative=0
                score_min=1000
                score_max=0
                score_average=0
                
                for k in $scores 
                do
                    if [[ $DEBUG_DISPLAY_RESULT -eq 1 ]] ; then
                        echo ----- | tee -a /log/3dmark/$DATE.brief.log
                        echo "current score: $k" | tee -a /log/3dmark/$DATE.brief.log
                        echo "score_cumulative: $scores_cumulative" | tee -a /log/3dmark/$DATE.brief.log
                        echo "score min/max: $score_min/$score_max" | tee -a /log/3dmark/$DATE.brief.log
                        if [[ $k<$score_min ]] ; then echo "min score found: $k" | tee -a /log/3dmark/$DATE.brief.log ; score_min=$k ; fi
                        if [[ $k>$score_max ]] ; then echo "max score found: $k" | tee -a /log/3dmark/$DATE.brief.log; score_max=$k ; fi
                    fi
                    scores_cumulative=`bc -l <<< $scores_cumulative+$k`
                    
                done
                score_average=`bc -l <<< $scores_cumulative/$scores_count`
                echo "average:score:: $score_average" | tee -a /log/3dmark/$DATE.brief.log

                if [[ $DEBUG_DISPLAY_RESULT -eq 1 ]] ; then
                    echo "max: $score_max" | tee -a /log/3dmark/$DATE.brief.log
                    echo "min: $score_min" | tee -a /log/3dmark/$DATE.brief.log
                fi
            done
        done
    else
        echo "Does not support displaying result for $GAME"
    fi
}

#       Copy game files from $REPO_SERVER_IP:/$REPO_SERVER_LOCATION
#       input:
#       $1 - name of directory in $REPO_SERVER_LOCATION to copy

function copy_game_files() {
    game_dir_src=$1
    game_dir_dest=$2

    if [[ -z $game_dir_src ]] ; then
            echo "Error: need to specify the game in p1"
            exit 1
    fi

    if [[ -z $game_dir_dest ]] ; then
            game_dir_dest="."
    fi

    if [[ $CONFIG_BYPASS_SETUP_REPO_SERVER -eq 0 ]] ; then
        set_repo_server
    fi
    
    echo "Destination path: $game_dir_dest"
    sudo mkdir -p $game_dir_dest

    if [[ ! "$(ls -A $game_dir_dest)" ]] ; then
        echo "$game_dir_dest does not exist."
        sudo mkdir -p $game_dir_dest
        echo "Copying $game_dir_src from $REPO_SERVER_IP, will take some time..."

        if [[ $OPTION_FILE_COPY_PROTOCOL == $FILE_COPY_RSYNC ]] ; then
            echo 'sudo sshpass -p amd1234 rsync -v -z -r -e "ssh -o StrictHostKeyChecking=no" root@$REPO_SERVER_IP:/$REPO_SERVER_LOCATION/$game_dir_src/* $game_dir_dest'
            echo "REPO_SERVER_LOCATION: $REPO_SERVER_LOCATION/"
            echo "game_dir_dest: $game_dir_dest"
            echo "game_dir_src: $game_dir_src"
            sudo sshpass -p amd1234 rsync -v -z -r -e "ssh -o StrictHostKeyChecking=no" root@$REPO_SERVER_IP:/$REPO_SERVER_LOCATION/$game_dir_src/* $game_dir_dest
        elif [[ $OPTION_FILE_COPY_PROTOCOL == $FILE_COPY_SCP ]] ; then
            echo "REPO_SERVER_LOCATION: $REPO_SERVER_LOCATION/"
            echo "game_dir_dest: $game_dir_dest"
            echo "game_dir_src: $game_dir_src"
            echo 'sudo sshpass -p amd1234 scp -C -v -r -o StrictHostKeyChecking=no root@$REPO_SERVER_IP:$REPO_SERVER_LOCATION/$game_dir_src/* ~/$game_dir_dest/'
            sudo sshpass -p amd1234 scp -C -v -r -o StrictHostKeyChecking=no root@$REPO_SERVER_IP:$REPO_SERVER_LOCATION/$game_dir_src/* ~/$game_dir_dest/
        else
            echo "ERROR: Unknown or unsupported copy protocol."
        fi

        if [[ $? -ne 0 ]] ; then
            echo "Failed to copy $game_dir_src..."
            exit 1
        fi
    else
        echo "$game_dir_dest exists, skipping."
    fi
}

# set render and encoder resolution.

function set_resolution() {
    pResolution=$1
    pGame=$2
    resoW=( 1280 1920 3840 )    
    resoH=( 720 1080 2160 )
    resoRender=( 720p 1080p 4k )
    resoHset=""
    resoWset=""

    if [[ -z $pResolution ]] ; then
        echo "resolution is empty. Setting to default 1080."
        CONFIG_RESOLUTION=$RESOLUTION_1080
	export 	GGP_VK_PRIMARY_SURFACE_EXTENT=1080p
    else
        counter=0
        for i in ${RESOLUTIONS_SUPPORTED[@]}
        do 
            if [[ $i == $pResolution ]] ; then
                CONFIG_RESOLUTION=$pResolution
                resoHset=${resoH[$counter]}
                resoWset=${resoW[$counter]}
                sudo sed -i "/encode_width/c \ \encode_width: $resoWset" $CONFIG_POLICY_DIR/$STREAMER_POLICY_FILE
                sudo sed -i "/encode_height/c \ \encode_height: $resoHset" $CONFIG_POLICY_DIR/$STREAMER_POLICY_FILE

                if [[ $pGame == GAME_3DMARK ]] ; then
                    echo "Setting json for 3dmark too..."
                    sudo sed -i "/resolution/c \ \"resolution" : $resoWsetx$resoHset," ../../configs/gt1.json
                    sudo sed -i "/resolution/c \ \"resolution" : $resoWsetx$resoHset," ../../configs/gt2.json
                fi

		export GGP_VK_PRIMARY_SURFACE_EXTENT=${resoRender[$counter]}
                sleep  3
                break
            fi
            counter=$((counter+1))
        done    

    fi    

    if [[ -z $CONFIG_RESOLUTION ]] ; then
        echo "Unable to set the resolution! Defaulting to 1080p"
        CONFIG_RESOLUTION=$RESOLUTION_1080
        sleep 30
    fi
    readBackRenderReso=`env  | grep GGP_VK_PRIMARY_SURFACE_EXTENT`
    echo "render resolution set to: $readBackRenderReso."
}
