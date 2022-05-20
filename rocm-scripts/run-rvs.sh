ROCM_ROOT=/opt/rocm-5.0.0
RVS=$ROCM_ROOT/rvs/rvs
RVS_CONFIG_ROOT=$ROCM_ROOT/rvs/conf/
LOG_FOLDER=log
mkdir $LOG_FOLDER -p

if [[ ! -f $RVS ]] ; then
    echo "Unable to find rvs in $RVS."
    ls -l $RVS
    exit 1
fi

for i in gst_single.conf pqt_single.conf pebb_single.conf babel.conf gpup_single.conf  ; do
    dmesg --clear
    $RVS -c $RVS_CONFIG_ROOT/$i 2>&1 | sudo tee $LOG_FOLDER/rvs.gpu.$i.log
    dmesg | sudo tee $LOG_FOLDER/rvs.gpu.$i.dmesg.log
done
