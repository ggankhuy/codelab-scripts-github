clear
echo Setup Yeti system for 3dmark on ubuntu 1604 / 1803...
sleep 3
sudo mkdir -p /opt/cloudcast/lib
sudo ln -s /opt/amdgpu-pro/lib/x86_64-linux-gnu/amdvlk64.so /opt/cloudcast/lib/amdvlk64.so
sudo mkdir -p /usr/local/cloudcast/
sudo ln -s ~/yeti-eng-bundle/lib /usr/local/cloudcast/lib
mkdir -p ~/.local/share/vulkan/icd.d
cp ~/yeti-eng-bundle/etc/vulkan/icd.d/yetivlk.json ~/.local/share/vulkan/icd.d/
sudo mkdir -p /usr/local/cloudcast/etc/yetivlk
sudo cp ~/yeti-eng-bundle/etc/yetivlk/config.json /usr/local/cloudcast/etc/yetivlk
