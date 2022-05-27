
DATE_FIXED=`date +%Y%m%d-%H-%M-%S`
ROCM_ROOT=/opt/rocm-5.1.3
RVS=$ROCM_ROOT/rvs/rvs
RVS_CONFIG_ROOT=$ROCM_ROOT/rvs/conf/
LOG_FOLDER_BASE=log/rvs-log
LOG_SUMMARY=$LOG_FOLDER/summary.log

echo "start" | sudo tee $LOG_SUMMARY
if [[ ! -f $RVS ]] ; then
    echo "Unable to find rvs in $RVS."
    ls -l $RVS
    exit 1
fi

echo "START..." >> $LOG_SUMMARY

#for i in gst_single.conf pqt_single.conf pebb_single.conf babel.conf gpup_single.conf  ; do
for i in {1..14} ; do
    FILE=pqt_single-$i.conf
    DATE=`date +%Y%m%d-%H-%M-%S`

    LOG_FOLDER=$LOG_FOLDER_BASE/$DATE_FIXED/$FILE
    mkdir $LOG_FOLDER -p
    dmesg --clear
    echo $DATE | sudo tee -a $LOG_SUMMARY
    echo $RVS -c $i | sudo tee -a $LOG_SUMMARY
    $RVS -d 4  -c $RVS_CONFIG_ROOT/$FILE 2>&1 | sudo tee -a $LOG_FOLDER/rvs.gpu.action-$i.log
    dmesg | sudo tee $LOG_FOLDER/rvs.gpu.$i.dmesg.log
    sleep 30
done
