ROCM_ROOT=/opt/rocm-4.5.2/
RVS=$ROCM_ROOT/rvs/rvs
RVS_CONFIG_ROOT=$ROCM_ROOT/rvs/conf/
LOG_FOLDER=log
mkdir $LOG_FOLDER -p
LOG_SUMMARY=$LOG_FOLDER/summary.log

echo "start" | sudo tee $LOG_SUMMARY
for i in gst_single.conf pqt_single.conf pebb_single.conf babel.conf gpup_single.conf  ; do
    dmesg --clear
    DATE=`date +%Y%m%d-%H-%M-%S`
    echo $DATE | sudo tee -a $LOG_SUMMARY
    echo $RVS -c $i | sudo tee -a $LOG_SUMMARY
    $RVS -c $RVS_CONFIG_ROOT/$i 2>&1 | sudo tee -a $LOG_FOLDER/rvs.gpu.$i.log
    dmesg | sudo tee $LOG_FOLDER/rvs.gpu.$i.dmesg.log
    sleep 300
done
