import os 
os.popen("apt install tree -y").read()
cmds=[ "uname -r", "ls -l /boot", "ls -l /usr/src/", "ls -l /lib/modules/", \
"tree -f /usr/src/ | grep -i arcturus", "find /usr/src/ -name amdgpu_amdkfd_arcturus.c", \
"tree -f /lib/modules/ | grep -i arcturus", "modinfo amdgpu | egrep \"filename|version|arcturus\"", \
"dpkg -l | grep amdgpu | grep firmware", "dpkg-query -L amdgpu-dkms-firmware | grep arcturus"]

for i in cmds:
	print("=============")
	print(i)
	ret=os.popen(i).read()
	print("-----------------")
	print(ret)


