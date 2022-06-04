p1=$1
modprobe amdgpu
cwd=`pwd`
cd /work/vats/package_script/run/ubuntu
chmod 755 yeti_environment.sh
./yeti_environment.sh /work/ubuntu_guest_package/utilities/test-apps/yeti
chmod 755 vk_example.sh

if [[ -z $p1 ]] ; then
    echo "Running infinite number of times..."
    i=0; while true; do echo "vkexample: $i"; ./vk_example.sh /work/ubuntu_guest_package/utilities/test-apps/vk_examples/  >> $i.log ;((i++)); done
else
    echo "Running $p1 times..."
    for i in $(seq 1 $p1)
    do
        echo "vkexample: $i"; ./vk_example.sh /work/ubuntu_guest_package/utilities/test-apps/vk_examples/  >> $cwd/$i.log
    done
fi
