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

GIM_LOC_DST=`modinfo gim | grep filename | tr -s ' ' | cut -d ' ' -f2`

if [[ OPTION_LIBGV != 0 ]] ; then
	GIM_API_LOC=`modinfo gim_api | grep filename | tr -s ' ' | cut -d ' ' -f2`
fi

PWD=`pwd`
GIM_LOC_SRC=$PWD/sriov_drv

#	For libgv build and install. (Excludes gim-api).
#	Set 1 to build install libgv.
#	Set 0 to build install gim.

OPTION_LIBGV=1

modprobe -r gim 

if [[ $? -ne 0 ]] ; then 
	echo "Failed to unload!!! Are VM-s running?"
	exit 1
fi

if [[ OPTION_LIBGV -eq 0 ]] ; then
	modprobe -r  gim-api
	cd  $PWD/gim-api
	make 
	cd ..
	cp $PWD/gim-api/gim-api.ko $GIM_API_LOC
	echo ---------------------------------------------------------------------
	echo checksum $PWD/gim-api/gim-api.ko: `md5sum $PWD/gim-api/gim-api.ko`
	echo checksum $GIM_API_LOC: `md5sum $GIM_API_LOC`
elif [[ OPTION_LIBGV -eq 1 ]] ; then
	GIM_LOC_SRC=$PWD/
else
	echo "Error! OPTION_LIBGV shouhld either be set to 1 or 0. Currently set to: $OPTION_LIBGV"
	exit 1
fi

cd $GIM_LOC_SRC
make

cd ..
cp $GIM_LOC_SRC/gim.ko $GIM_LOC_DST

echo ---------------------------------------------------------------------
echo checksum $GIM_LOC_SRC/gim.ko: `md5sum $GIM_LOC_SRC/gim.ko`
echo checksum $GIM_LOC_DST: `md5sum $GIM_LOC_DST`
echo ---------------------------------------------------------------------

dmesg --clear

if [[ OPTION_LIBGV != 0 ]] ; then
	modprobe gim-api
fi

modprobe gim
dmesg | egrep -i "production|release"
