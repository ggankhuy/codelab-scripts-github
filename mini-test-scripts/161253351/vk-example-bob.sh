cd /work/vats/package_script/run/ubuntu
chmod 755 yeti_environment.sh
./yeti_environment.sh /work/ubuntu_guest_package/utilities/test-apps/yeti
chmod 755 vk_example.sh
i=0; while true; do echo "vkexample: $i"; ./vk_example.sh /work/ubuntu_guest_package/utilities/test-apps/vk_examples/  >> $i.log ;((i++)); done
