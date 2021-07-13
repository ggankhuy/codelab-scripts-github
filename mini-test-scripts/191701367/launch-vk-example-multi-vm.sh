WAIT_INTERVAL=5
CONFIG_VATS2_SUPPORT=1
CONFIG_ITERATIONS=50
CONFIG_BY_PASS_XGEMM=0
CONFIG_BARRIER_USE_WAIT=1 
CONFIG_DMESG_ONLY=0

DATE=`date +%Y%m%d-%H-%M-%S`
DIRNAME=191701367-result/$DATE/
mkdir -p $DIRNAME
p0=$0
p1=$1
p2=$2
SINGLE_BAR="---------------------------------------------"
DOUBLE_BAR="============================================"
if [[ $p1 == "--help" ]] ; then
	clear
	echo "Usage: "
	echo "$p0 --help: display this help."
	echo "$p0 --all: run on all VM"
	echo "$p0 <VM_NAME>: run on specified VM <VM_NAME>"
	exit 0
fi

if [[ -z $p1 ]] ; then
	echo "VM name not specified, defaulting to vats-test-01..."
	VM_NAME=vats-test-01
else
	echo "VM name: $p1..."
	sleep 5
	VM_NAME=$p1
fi

if [[ $p1 -eq "all" ]] ; then
	if [[ $CONFIG_VATS2_SUPPORT -eq 1 ]] ; then
	    VM_GREP_PATTERN=vats
	else
	    VM_GREP_PATTERN=gpu
	fi
fi

echo "Backup host dmesg as dmesg-host-$DATE.log..."
dmesg >$DIRNAME/dmesg-host-$DATE.log
dmesg --clear
TOTAL_VMS=`virsh list --all | grep -i $VM_GREP_PATTERN | grep running | wc -l`

# limiting total vms.

TOTAL_VMS=8
echo "TOTAL_VMS: $TOTAL_VMS"

if [[ $CONFIG_DMESG_ONLY -ne 1 ]] ; then
    nohup ./monitor.sh &
    sleep 10
fi

for (( n=0; n < $TOTAL_VMS; n++ ))  ; do
    echo $DOUBLE_BAR
    echo n: $n
    GPU_INDEX=$n
    VM_INDEX=$(($n+1))
    echo "VM_INDEX: $VM_INDEX"

    VM_NAME=`virsh list --all |         grep $VM_GREP_PATTERN | head -$(($GPU_INDEX+1)) | tail -1  | tr -s ' ' | cut -d ' ' -f3`
    VM_NAMES[$n]=$VM_NAME
    VM_NO=`virsh list --all |         grep $VM_GREP_PATTERN | head -$(($GPU_INDEX)) | tail -1  | tr -s ' ' | cut -d ' ' -f2`
    VM_IP=`virsh domifaddr $VM_NAME | grep ipv4 | tr -s ' ' | cut -d ' ' -f5 | cut -d '/' -f1`
    VM_IPS[$n]=$VM_IP

    echo VM_NAME: $VM_NAME, VM_INDEX: $VM_INDEX, VM_NO: $VM_NO, GPU_INDEX: $GPU_INDEX, VM_IP: $VM_IP
    sleep 1

    if [[ $CONFIG_DMESG_ONLY -ne 1 ]] ; then
        sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$VM_IP 'cd /work/ubuntu_guest_package/utilities/test-apps/vk_examples/build_for_gibraltar ; dmesg --clear ; modprobe amdgpu ; rm *.log ; for i in {0..10} ; do ./run_vk_examples.sh ; done' > $DIRNAME/tmp.$n.log &
    fi
done

echo "dmesg loop."

i=0
while true
do
    echo iter $i:
    for (( n=0; n < $TOTAL_VMS; n++ ))  ; 
    do
        sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$VM_IP 'dmesg' > $DIRNAME/dmesg.vm-$n.loop.$i.log &
    done
    i=$((i+1))
    sleep 300
    rm  monitor.log
done

