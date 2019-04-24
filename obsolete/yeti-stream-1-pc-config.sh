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
