cd /work/vats/package_script/run/ubuntu
chmod 755 vk_example.sh
i=0; while true; do echo "vkexample: $i"; ./vk_example.sh /work/ubuntu_guest_package/utilities/test-apps/vk_examples/  >> 1.log ;((i++)); done
