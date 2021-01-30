# tested on ubuntu1804 only.
# https://vulkan.lunarg.com/doc/view/1.1.126.0/linux/getting_started.html
# source: https://vulkan.lunarg.com/doc/sdk/1.2.162.1/linux/getting_started_ubuntu.html

for i in \
"apt-get update" "apt-get dist-upgrade -y" \
"sudo apt-get install libglm-dev cmake libxcb-dri3-0 libxcb-present0 libpciaccess0 libpng-dev libxcb-keysyms1-dev libxcb-dri3-dev libx11-dev g++ gcc g++-multilib libmirclient-dev libwayland-dev libxrandr-dev libxcb-ewmh-dev git python3 bison -y" \
"apt-get install qt5-default qtwebengine5-dev -y" "apt update" \
'wget -qO - http://packages.lunarg.com/lunarg-signing-key-pub.asc | apt-key add -' \
'wget -qO /etc/apt/sources.list.d/lunarg-vulkan-bionic.list http://packages.lunarg.com/vulkan/lunarg-vulkan-bionic.list' \
"apt update" "apt install vulkan-sdk -y" \
"which vkvia" "vkvia vulkaninfo" "vkcube" ;
do
	echo --------------------
	echo "sh: Executing $i"
	$i
done
