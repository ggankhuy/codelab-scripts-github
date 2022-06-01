vmIp=10.6.189.11
pw=val
user=root
for i in {1..10} ; do
    sshpass -p $pw ssh -o StrictHostKeyChecking=no $user@$vmIp 'cd /root/gg/ad-hoc-scripts/rocm-scripts; ./run-rvs.sh'
    sshpass -p $pw ssh -o StrictHostKeyChecking=no $user@$vmIp 'reboot'
done
