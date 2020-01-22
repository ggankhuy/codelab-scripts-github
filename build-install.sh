#	Use this from the root folder of gim source.
#	If it is run from other location, result is undefined and will fail.
#	- unloads current gim-api, gim
#	- build  gim-api.ko and gim.ko from gim-api and sriov_drv folder, respectively.
#	- cp built modules from current location to kernel library location (usually: /lib/modules)
#	using modinfo output.
#	- reload the newly built modules.
#	- provide checksum for copy verification.
#	- check dmesg signature for line containing release.

#for i in {1..4} ; do virsh shutdown debian-drop-2019-q3-rc7-gpu$i-vf00 ; done;

OPTION_LIBGV=1

#   This does not work when no gim is loaded or present before. Keep it just in case.
#GIM_LOC_DST=`modinfo gim | grep filename | tr -s ' ' | cut -d ' ' -f2`

GIM_LOC_DST=/lib/modules/`uname -r`/kernel/drivers/sriov_drv/
GIM_API_LOC_DST=/lib/modules/`uname -r`/kernel/drivers/gim-api/

mkdir -p $GIM_LOC_DST
mkdir -p $GIM_API_LOC_DST

echo "GIM_LOC_DST: $GIM_LOC_DST"
echo "GIM_API_LOC_DST: $GIM_API_LOC_DST"

if [[ OPTION_LIBGV == 0 ]] ; then
	GIM_API_LOC=`modinfo gim_api | grep filename | tr -s ' ' | cut -d ' ' -f2`
fi

PWD=`pwd`
GIM_LOC_SRC=$PWD/sriov_drv
GIM_API_LOC_SRC=$PWD/gim_api

#	For libgv build and install. (Excludes gim-api).
#	Set 1 to build install libgv.
#	Set 0 to build install gim.

modprobe -r gim 

if [[ $? -ne 0 ]] ; then 
	echo "Failed to unload!!! Are VM-s unning?"
	#exit 1
fi

if [[ OPTION_LIBGV -eq 0 ]] ; then
	modprobe -r  gim-api
	cd  $PWD/gim-api
	make 
	cd ..
	echo cp $PWD/gim-api/gim-api.ko $GIM_API_LOC_DST
	cp $PWD/gim-api/gim-api.ko $GIM_API_LOC_DST
	echo ---------------------------------------------------------------------
	echo checksum $PWD/gim-api/gim-api.ko: `md5sum $PWD/gim-api/gim-api.ko`
	echo checksum $GIM_API_LOC_DST: `md5sum $GIM_API_LOC_DST/gim-api.ko`
elif [[ OPTION_LIBGV -eq 1 ]] ; then
	GIM_LOC_SRC=$PWD/
else
	echo "Error! OPTION_LIBGV shouhld either be set to 1 or 0. Currently set to: $OPTION_LIBGV"
	exit 1
fi

cd $GIM_LOC_SRC
make

cd ..
echo cp $GIM_LOC_SRC/gim.ko $GIM_LOC_DST
cp $GIM_LOC_SRC/gim.ko $GIM_LOC_DST

echo ---------------------------------------------------------------------
echo checksum $GIM_LOC_SRC/gim.ko: `md5sum $GIM_LOC_SRC/gim.ko`
echo checksum $GIM_LOC_DST: `md5sum $GIM_LOC_DST/gim.ko`
echo ---------------------------------------------------------------------

dmesg --clear

if [[ OPTION_LIBGV == 0 ]] ; then
	modprobe gim-api
fi

dmesg | egrep -i "production|release"
depmod
modprobe gim
lsmod | grep gim
