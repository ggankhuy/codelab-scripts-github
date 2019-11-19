#	Use this from the root folder of gim source.
#	If it is run from other location, result is undefined and will fail.
#	- unloads current gim-api, gim
#	- build  gim-api.ko and gim.ko from gim-api and sriov_drv folder, respectively.
#	- cp built modules from current location to kernel library location (usually: /lib/modules)
#	using modinfo output.
#	- reload the newly built modules.
#	- provide checksum for copy verification.
#	- check dmesg signature for line containing release.

#for i in {0..7} ; do virsh shutdown debian-drop-2019-q3-rc7-gpu$i-vf00 ; done;

GIM_LOC=`modinfo gim | grep filename | tr -s ' ' | cut -d ' ' -f2`
GIM_API_LOC=`modinfo gim_api | grep filename | tr -s ' ' | cut -d ' ' -f2`
PWD=`pwd`

modprobe -r gim 
modprobe -r  gim-api

cd  $PWD/gim-api
make 
cd ..
cd $PWD/sriov_drv
make

cd ..
cp $PWD/gim-api/gim-api.ko $GIM_API_LOC
cp $PWD/sriov_drv/gim.ko $GIM_LOC

echo ---------------------------------------------------------------------
echo checksum $PWD/gim-api/gim-api.ko: `md5sum $PWD/gim-api/gim-api.ko`
echo checksum $GIM_API_LOC: `md5sum $GIM_API_LOC`
echo ---------------------------------------------------------------------
echo checksum $PWD/sriov_drv/gim.ko: `md5sum $PWD/sriov_drv/gim.ko`
echo checksum $GIM_LOC: `md5sum $GIM_LOC`
echo ---------------------------------------------------------------------

dmesg --clear
modprobe gim-api
modprobe gim

dmesg | egrep -i "production|release"
