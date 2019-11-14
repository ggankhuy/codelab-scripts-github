GIM_LOC=modinfo gim | grep filename | tr -s ' ' | cut -d ' ' -f2
GIM_API_LOC=modinfo gim_api | grep filename | tr -s ' ' | cut -d ' ' -f2
cd  gim-api
make 
cp ./gim-api.ko  $GIM_API_LOC
cd ..
cd gim
make
cp ./gim.ko $GIM_LOC
for i in {0..7} ; do virsh shutdown debian-drop-2019-q3-rc7-gpu$i-vf00 ; done;
modprobe -r gim 
modprobe -r  gim-api
dmesg --clear
modprobe gim-api
modprobe gim

