#   Following path definitions are VATS2 configured VM-s. It will differ for VATS1 and result likely be unpredictable.

#   
CONFIG_PATH_XGEMM=/work/ubuntu_guest_package/utilities/test-apps/xgemm/SgemmStressTest/
CONFIG_PATH_XGEMM_BINARY=/work/ubuntu_guest_package/utilities/test-apps/xgemm/
CONFIG_FILENAME_XGEMM_OUTPUT=PerfoGemm_GPU_0.csv
CONFIG_FILENAME_XGEMM_FIND_MAX=./xgemm-find-max.py
CONFIG_FILENAME_ATITOOL=/root/tools/atitool/atitool
CONFIG_OUTPUT_DIR=./xgemm_out
p1=$1
p0=$0
CONFIG_GPU_INDEX=$2
CONFIG_IP_GUEST=$p1

DATE=`date +%Y%m%d-%H-%M-%S`

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

if [[ -z $CONFIG_GPU_INDEX ]] ; then
    echo "Need GPU index for atitool. It must match the VM whose IP specified in p1. Otherwise pmlog will capture data from wrong VM..."
    exit 1
fi

if [[ ! -f $CONFIG_FILENAME_XGEMM_FIND_MAX ]] ; then
    echo "Unable to find $CONFIG_FILENAME_XGEMM_FIND_MAX"
    exit 1
fi

if [[ ! -f $CONFIG_FILENAME_ATITOOL ]] ; then
    echo "atitool is not available as path: $CONFIG_FILENAME_ATITOOL"
    exit 1
fi

#   Start capturing PM log:

echo  "Start capturing PM log from i=$CONFIG_GPU_INDEX..."
#echo $CONFIG_FILENAME_ATITOOL -pmoutput $CONFIG_OUTPUT_DIR/PMLOG-$DATE.csv -pmlogall -i=$CONFIG_GPU_INDEX
#$CONFIG_FILENAME_ATITOOL -pmoutput=$CONFIG_OUTPUT_DIR/PMLOG-$DATE.csv -pmlogall -i=$CONFIG_GPU_INDEX -pmcount=20 &
$CONFIG_FILENAME_ATITOOL -pmoutput=$CONFIG_OUTPUT_DIR/PMLOG-$DATE.csv -pmlogall -i=$CONFIG_GPU_INDEX  &
PID_ATITOOL=$!
echo "Atitool PID: $PID_ATITOOL"
echo "Idle run for few seconds..."
sleep 10
echo "Launching xgemm on guest $CONFIG_IP_GUEST..."
echo "Guest VM IP:"  $CONFIG_IP_GUEST

for cmd in "cp $CONFIG_PATH_XGEMM_BINARY/xgemmStandaloneTest_NoCPU $CONFIG_PATH_XGEMM" "modprobe amdgpu" \
"cd $CONFIG_PATH_XGEMM ; chmod 755 xgemmStandaloneTest_NoCPU ; pwd ; sleep 3; ./xgemmStandaloneTest_NoCPU" 
do 
    sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$CONFIG_IP_GUEST $cmd
done

mkdir $CONFIG_OUTPUT_DIR
sshpass -p amd1234 scp root@$CONFIG_IP_GUEST:/$CONFIG_PATH_XGEMM/$CONFIG_FILENAME_XGEMM_OUTPUT $CONFIG_OUTPUT_DIR/

echo "Idle run for few seconds before killing ..."
sleep 10
kill $PID_ATITOOL

./$CONFIG_FILENAME_XGEMM_FIND_MAX $CONFIG_OUTPUT_DIR/$CONFIG_FILENAME_XGEMM_OUTPUT | tee $CONFIG_OUTPUT_DIR/output_final.log
echo "output of pmlog: $CONFIG_OUTPUT_DIR/PMLOG-$DATE.csv"



