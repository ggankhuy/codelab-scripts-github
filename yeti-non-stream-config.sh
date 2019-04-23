clear
echo setting up Yeti libraries...
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
