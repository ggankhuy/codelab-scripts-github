#!/bin/bash

modprobe amdgpu
LOG_FOLDER=./log/pcie-link-down/
mkdir -p $LOG_FOLDER
IP_BMC=10.96.30.225
TEST_NAME="LINK down"

#pci_up_port=0000:2f:00.0
#pci_up_port=0000:2e:10.0
pci_up_port=0000:20:03.1
#echo "Capturing rocm_techsupport before " $TEST_NAME
#../rocm_techsupport.sh > $LOG_FOLDER/before_rocm_techsupport_$pci_bus_number.log

#echo "Capturing SEL logs after $TEST_NAME on $pci_bus_number"
#ipmitool -H $IP_BMC -U ADMIN -I lanplus -P ADMIN sel elist > after_SEL_logs_$pci_bus_number.log

echo "Setting link down in pcie cap + link control Rx..."
setpci -s $pci_up_port 3e.L=00000040
lspci -s $pci_up_port  -vvvxxx
echo ------------------------
lspci -s $pci_up_port  -vvvxxx | grep 30:
lspci -s $pci_up_port  -vvvxxx | grep BridgeCtl

#echo "Capturing rocm_techsupport after SERR injection on $pci_bus_number"
#../rocm_techsupport.sh > $LOG_FOLDER/after_rocm_techsupport_$pci_bus_number.log

#echo "Capturing SEL logs after SERR injection on $pci_bus_number"
#ipmitool -H $IP_BMC -U ADMIN -I lanplus -P ADMIN sel elist > after_SEL_logs_$pci_bus_number.log

#lspci | grep Disp
#USE_HIP_CALL=1 /root/rccl/tools/TransferBench/TransferBench example.cfg
