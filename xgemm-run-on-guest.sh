#   Following path definitions are VATS2 configured VM-s. It will differ for VATS1 and result likely be unpredictable.

#   
CONFIG_PATH_XGEMM=/work/ubuntu_guest_package/utilities/test-apps/xgemm/SgemmStressTest
CONFIG_PATH_XGEMM_BINARY=/work/ubuntu_guest_package/utilities/test-apps/xgemm/
CONFIG_FILENANE_XGEMM_OUTPUT=PerfoGemm_GPU_0.csv
CONFIG_FILENAME_XGEMM_FIND_MAX=./xgemm-find-max.py
CONFIG_FILENAME_ATITOOL=/root/tools/atitool/atitool
CONFIG_OUTPUT_DIR=./xgemm_out
CONFIG_GPU_INDEX=$2
CONFIG_IP_GUEST=$1

DATE=`date +%Y%m%d-%H-%M-%S`

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

if [[ !-z $CONFIG_FILENAME_ATITOOL ]] ; then
    echo "atitool is not available as path: $CONFIG_FILENAME_ATITOOL"
    exit 1
fi

#   Start capturing PM log:

$CONFIG_FILENAME_ATITOOL -pmoutput $CONFIG_OUTPUT_DIR/PMLOG-$DATE.csv -pmlogall -i=$CONFIG_GPU_INDEX

echo "Guest VM IP:"  $CONFIG_IP_GUEST

for cmd in 'cp $CONFIG_PATH_XGEMM_BINARY/xgemmStandaloneTest_NoCPU $CONFIG_PATH_XGEMM' 'chmod 755 $CONFIG_PATH_XGEMM/xgemmStandaloneTest_NoCPU'  \
do 
    echo "ssh root@$CONFIG_IP_GUEST $cmd "
done

mkdir $CONFIG_OUTPUT_DIR
scp root@$CONFIG_IP_GUEST:/$CONFIG_PATH_XGEMM/$CONFIG_FILENAME_XGEMM_OUTPUT $CONFIG_OUTPUT_DIR/


    

