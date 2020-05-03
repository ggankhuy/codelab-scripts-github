if [[ -z '$1'  ]] ; then echo "Usage: $0 <ip_address of the server to which rocm to be installed." ; exit 1 ; fi

VM_IP=$1

for i in "apt remove amdgpu-dkms" "apt update" "apt dist-upgrade -y" "apt install libnuma-dev -y " "echo rebooting ; sleep 15 ; reboot" ; do
        sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$VM_IP $i
done

sleep 15
for i in "wget -qO - http://repo.radeon.com/rocm/apt/debian/rocm.gpg.key | sudo apt-key add -" \
        "echo 'deb [arch=amd64] http://repo.radeon.com/rocm/apt/debian/ xenial main' | sudo tee /etc/apt/sources.list.d/rocm.list" \
        "apt update" "apt install rocm-dkms -y" ; do
        sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$VM_IP $i
done

