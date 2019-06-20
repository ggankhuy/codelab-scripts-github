#  this script assumes all the VMs on either ixt39 or ixt70 is created afresh usign scripts i.e.
#  runtest 6 from autotest.
#  
#  count all vms.
#  load gim.
#  start default network.
#  turn on all vms.
#  log on to each vm through ssh (determine ip using virsh domifaddr <vmno>
#  update /etc/network/interfaces with static ip from pool.
#  ip address range:  10.216.66.67-78.
#  assignment:
#  ixt39  4vm-s / 4 gpu-s.
#  ixt70  8vm-s / 8 gpu-s.


