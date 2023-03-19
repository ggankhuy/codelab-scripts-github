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
