if [[ -z '$1'  ]] ; then echo "Usage: $0 <ip_address of the server to which rocm to be installed." ; exit 1 ; fi

VM_IP=$1

for i in "apt remove amdgpu-dkms -y" "apt update -y" "apt dist-upgrade -y" "apt install libnuma-dev -y " "echo rebooting ; sleep 15 ; reboot" ; do
        echo ----------
        echo $i
        echo ----------
        sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$VM_IP $i
done

sleep 15
for i in "wget -qO - http://repo.radeon.com/rocm/apt/debian/rocm.gpg.key | sudo apt-key add -" \
        "echo 'deb [arch=amd64] http://repo.radeon.com/rocm/apt/debian/ xenial main' | sudo tee /etc/apt/sources.list.d/rocm.list" \
        "apt update" "apt install rocm-dkms -y" "modprobe amdgpu" "/opt/rocm/bin/rocminfo" "apt install clinfo -y" "clinfo"; 
do
        echo ----------
        echo $i
        echo ----------
        sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$VM_IP $i
done

apt install ocl-icd-opencl-dev libopenblas-dev  -y
