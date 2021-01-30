# tested on ubuntu1804 only.
# source: https://vulkan.lunarg.com/doc/sdk/1.2.162.1/linux/getting_started_ubuntu.html

sudo apt-get update
sudo apt-get dist-upgrade
sudo apt-get install libglm-dev cmake libxcb-dri3-0 libxcb-present0 libpciaccess0 libpng-dev libxcb-keysyms1-dev libxcb-dri3-dev libx11-dev g++ gcc g++-multilib libmirclient-dev libwayland-dev libxrandr-dev libxcb-ewmh-dev git python3 bison -y
sudo apt-get install qt5-default qtwebengine5-dev
apt update
wget -qO - http://packages.lunarg.com/lunarg-signing-key-pub.asc | sudo apt-key add -
sudo wget -qO /etc/apt/sources.list.d/lunarg-vulkan-bionic.list http://packages.lunarg.com/vulkan/lunarg-vulkan-bionic.list
sudo apt update
sudo apt install vulkan-sdk
which vkvia
vkvia
vulkaninfo
vkcube
