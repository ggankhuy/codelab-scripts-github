dmesg --clear
modprobe amdgpu
dmesg >> dmesg.amdgpu.log
dmesg --clear
pushd /work/ubuntu_guest_package/utilities/debug-tool/debug
python3 hangdump.py start
cd ../quark
./quark tests/production/hang/hang_vm_gfx0_bad_cs_dispatch.lua  &
sleep 10
dmesg  >> dmesg.quark.log
dmesg --clear
cd ..
echo "calling umr..."
./umr -O halt_waves -wa
ls -ltr /var/log/Hang*
dmesg >> dmesg.umr.log
dmesg



