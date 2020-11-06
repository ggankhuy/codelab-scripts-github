CONFIG_UMR_USE_VATS2_PATH=0
DATE=`date +%Y%m%d-%H-%M-%S`
CWD=`pwd`
mkdir -p ./log/$DATE/
dmesg --clear
modprobe amdgpu
dmesg >> dmesg.amdgpu.log
dmesg --clear
CONFIG_LOG_QUARK=$CWD/log/$DATE/quark
CONFIG_LOG_UMR_WAVES=$CWD/log/$DATE/umr.waves
CONFIG_LOG_DMESG=$CWD/log/$DATE/dmesg

echo start >  $CONFIG_LOG_QUARK.log
echo  start > $CONFIG_LOG_UMR_WAVES.log
echo start > $CONFIG_LOG_DMESG.log

echo "Log files: $CONFIG_LOG_QUARK, $CONFIG_LOG_UMR_WAVES, $CONFIG_LOG_DMESG"

pushd /work/ubuntu_guest_package/utilities/debug-tool/debug
python3 hangdump.py start
cd ../quark
./quark tests/production/hang/hang_vm_gfx0_bad_cs_dispatch.lua 2>&1 | tee $CONFIG_LOG_QUARK.log &
cd ..
echo "calling umr..."

for (( i=0;i<5 ; i++ )) ; do
	which umr
	read -p "Press a key to launch umr to read waves...."
	echo "outputting to $CONFIG_LOG_UMR_WAVES.log"
	/usr/bin/umr -O halt_waves -wa gfx_0.0.0 2>&1 | tee $CONFIG_LOG_UMR_WAVES.$i.log
	dmesg 2>&1 | tee $CONFIG_LOG_DMESG.$i.log
	echo "done..."
	sleep 2
done



