source ./common.sh
CONFIG_MEMCAT_LOC=/root/memcat/
CONFIG_MEMCAT_PATH=$CONFIG_MEMCAT_LOC/amd_memcat.stripped

chmod 755  $CONFIG_MEMCAT_PATH
dpkg -i  $CONFIG_MEMCAT_LOC/grtev4-x86-runtimes_1.0-145370904_amd64.deb
cd $CONFIG_MEMCAT_LOC
modprobe amdkfd
modprobe amdgpu
#common_runtime_setup novce
#sudo chmod -R g=u /usr/local/cloudcast/
#sudo chmod -R o=u /usr/local/cloudcast/

for (( i=0 ; i < 100; i ++ )) 
do
	echo "memcat write $ith time..."
	$CONFIG_MEMCAT_LOC/amd_memcat.stripped --action write --byte 0x55
done
