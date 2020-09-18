#   Following path definitions are VATS2 configured VM-s. It will differ for VATS1 and result likely be unpredictable.

#   
VATS2_SUPPORT=1
CONFIG_XGEMM_GIB=1
CONFIG_PMLOG_CAPTURE=0

if [[ $CONFIG_XGEMM_GIB -eq 1 ]] ; then
	CONFIG_PATH_XGEMM_HOST=/xgemm
else
	if [[  $VATS2_SUPPORT -eq 1 ]] ; then
		CONFIG_PATH_XGEMM=/work/ubuntu_guest_package/utilities/test-apps/xgemm/SgemmStressTest/
		CONFIG_PATH_XGEMM_BINARY=/work/ubuntu_guest_package/utilities/test-apps/xgemm/
	else
		CONFIG_PATH_XGEMM=/work/drop*/test-apps/xgemm/SgemmStressTest
		CONFIG_PATH_XGEMM_BINARY=/work/drop*/test-apps/xgemm/
	fi
fi
CONFIG_FILENAME_XGEMM_OUTPUT=PerfoGemm_GPU_0.csv
CONFIG_FILENAME_XGEMM_FIND_MAX=./xgemm-find-max.py
CONFIG_FILENAME_ATITOOL=/root/tools/atitool/atitool
p1=$1
p0=$0

CONFIG_OUTPUT_DIR=./xgemm_out
CONFIG_GPU_INDEX=$2
CONFIG_IP_GUEST=$p1

DATE=`date +%Y%m%d-%H-%M-%S`
CONFIG_OUTPUT_DIR=./xgemm_out_$DATE
OUTPUT_SUMMARY=output_summary_$DATE.log

mkdir $CONFIG_OUTPUT_DIR

if [[ $p1 == "--help" ]] ; then
	clear
	echo ===========================
	echo "Usage: $p0  <GUEST_VM_IP> <GUEST_VM_IP_PF_INDEX>"
	echo ===========================
	exit 0
fi

if [[ -z $CONFIG_IP_GUEST ]] ; then
    echo "Need IP address for guest image..."
    exit 1
fi   

if [[ -z $CONFIG_GPU_INDEX ]] && [[ $CONFIG_PMLOG_CAPTURE -eq 1 ]] ; then
    echo "Need GPU index for atitool. It must match the VM whose IP specified in p1. Otherwise pmlog will capture data from wrong VM..."
    exit 1
fi

if [[ ! -f $CONFIG_FILENAME_XGEMM_FIND_MAX ]] ; then
    echo "Unable to find $CONFIG_FILENAME_XGEMM_FIND_MAX"
    exit 1
fi

if [[ ! -f $CONFIG_FILENAME_ATITOOL ]] && [[ $CONFIG_PMLOG_CAPTURE -eq 1 ]] ; then
    echo "atitool is not available as path: $CONFIG_FILENAME_ATITOOL"
    exit 1
fi

#   Start capturing PM log:

mkdir $CONFIG_OUTPUT_DIR

if [[ $CONFIG_PMLOG_CAPTURE -eq 1 ]] ; then
	echo  "Start capturing PM log from i=$CONFIG_GPU_INDEX..."
	$CONFIG_FILENAME_ATITOOL -pmoutput=$CONFIG_OUTPUT_DIR/PMLOG-$DATE.csv -pmlogall -i=$CONFIG_GPU_INDEX  &
	PID_ATITOOL=$!
	echo "Atitool PID: $PID_ATITOOL"
	echo "Idle run for few seconds..."
	sleep 10
fi

echo "Launching xgemm on guest $CONFIG_IP_GUEST..."
echo "Guest VM IP:"  $CONFIG_IP_GUEST

if [[ $CONFIG_XGEMM_GIB -eq 1 ]] ; then
	for cmd in "modprobe amdgpu; cp /xgemm/multi_amd_xgemm.stripped /xgemm/multi_amd_xgemm; dpkg -i /xgemm/grtev4-x86-runtimes_1.0-145370904_amd64.deb ; apt install ocl-icd-opencl-dev libopenblas-dev -y ; cd /xgemm; chmod 755 ./multi_amd_xgemm ;./multi_amd_xgemm --logtostderr --xgemm_kernel_compile_arguments="-cl-std=CL2.0" --noenforce_kernel_ipv6_support --iterations 100" 
	do 
	    sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$CONFIG_IP_GUEST $cmd
	    if [[ $? -ne 0 ]] ; then
		echo "Error executing $cmd, giving up..."

		if [[ $CONFIG_PMLOG_CAPTURE -eq 1 ]] ; then
			kill $PID_ATITOOL
		fi
		exit 1
	    fi
	done
else
	for cmd in "cp $CONFIG_PATH_XGEMM_BINARY/xgemmStandaloneTest_NoCPU $CONFIG_PATH_XGEMM" "modprobe amdgpu" \
	"cd $CONFIG_PATH_XGEMM ; chmod 755 xgemmStandaloneTest_NoCPU ; pwd ; sleep 3; ./xgemmStandaloneTest_NoCPU" 
	do 
	    sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$CONFIG_IP_GUEST $cmd
	    if [[ $? -ne 0 ]] ; then
		echo "Error executing $cmd, giving up..."

		if [[ $CONFIG_PMLOG_CAPTURE ]] ; then
			kill $PID_ATITOOL
		fi
		exit 1
	    fi
	done
fi
if [[ $CONFIG_XGEMM_GIB -ne 1 ]] ; then
	sshpass -p amd1234 scp root@$CONFIG_IP_GUEST:/$CONFIG_PATH_XGEMM/$CONFIG_FILENAME_XGEMM_OUTPUT $CONFIG_OUTPUT_DIR/
fi

echo "Idle run for few seconds before killing ..."
sleep 10

if [[ $CONFIG_PMLOG_CAPTURE -eq 1 ]] ; then
	kill $PID_ATITOOL
fi

if [[ $CONFIG_XGEMM_GIB -ne 1 ]] ; then
	./$CONFIG_FILENAME_XGEMM_FIND_MAX $CONFIG_OUTPUT_DIR/$CONFIG_FILENAME_XGEMM_OUTPUT | tee $CONFIG_OUTPUT_DIR/$OUTPUT_SUMMARY
	echo "output of pmlog: $CONFIG_OUTPUT_DIR/PMLOG-$DATE.csv"
fi

    

