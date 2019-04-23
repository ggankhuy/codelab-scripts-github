clear
echo setting up Yeti on client machine...

apt install -y libc++abi-dev
export LD_LIBRARY_PATH=~/yeti-eng-bundle/lib
cd ~/yeti-eng-bundle/bin
echo "Type, but do not execute the following command:"
echo "./game_client run-direct <IPv4 address of the Yeti computer>:44700"



