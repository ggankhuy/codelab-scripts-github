
DATE_FIXED=`date +%Y%m%d-%H-%M-%S`
ROCM_ROOT=/opt/rocm-5.2.0
RVS=$ROCM_ROOT/rvs/rvs
#RVS_CONFIG_ROOT=$ROCM_ROOT/rvs/conf/
LOG_FOLDER_BASE=log/rvs/
mkdir -p $LOG_FOLDER_BASE
LOG_SUMMARY=$LOG_FOLDER_BASE/summary.log
CONF_PATH=$ROCM_ROOT/share/rocm-validation-suite/conf/

echo "start" | sudo tee $LOG_SUMMARY
echo "`hostname`" | sudo tee -a $LOG_SUMMARY
if [[ ! -f $RVS ]] ; then
    echo "Unable to find rvs in $RVS."
    ls -l $RVS
    exit 1
fi

echo "START..." >> $LOG_SUMMARY

#for i in gst_single.conf pqt_single.conf pebb_single.conf babel.conf gpup_single.conf  ; do
#for i in gst_single.conf pebb_single.conf  ; do
for i in pqt_single_2_devices_manoj.conf  ; do
    for j in {1..10} ; do
        FILE=$i-$j
        DATE=`date +%Y%m%d-%H-%M-%S`

        LOG_FOLDER=$LOG_FOLDER_BASE/$DATE_FIXED/$FILE/
        mkdir $LOG_FOLDER -p
        dmesg --clear
        echo $DATE | sudo tee -a $LOG_SUMMARY
        echo $RVS -c $i | sudo tee -a $LOG_SUMMARY
        $RVS -d 4  -c $CONF_PATH/$i 2>&1 | sudo tee -a $LOG_FOLDER/rvs.gpu.action-$i.log
        dmesg | sudo tee $LOG_FOLDER/rvs.gpu.$i.dmesg.log
        sleep 30
    done
done
