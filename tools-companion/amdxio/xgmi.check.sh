for var in "$@"
do

    if [[ ! -z `echo "$var" | grep "install"` ]]  ; then
        PATH=`echo $var | cut -d '=' -f2`
        
        echo "cd `pwd`; `pwd`/$0" >> /etc/rc.local
        exit 0
    fi
done
AMDXIO_PATH=.
LOG_PATH=/log/xgmi-stat/
DATE=`date +%Y%m%d-%H-%M-%S`
if [[ ! -f $AMDXIO_PATH/AMDXIO ]] ; then 
    echo "Can not find AMDXIO path: $AMDXIO_PATH/AMDXIO"
    exit 1
fi
LOG_PATH_CURR_BOOT=$LOG_PATH/$DATE
mkdir -p /$LOG_PATH_CURR_BOOT
LOG_FILE_BEFORE_LOAD=$LOG_PATH_CURR_BOOT/amdxio-xgmistatus-before-driver-load
LOG_FILE_AFTER_LOAD=$LOG_PATH_CURR_BOOT/amdxio-xgmistatus-after-driver-load

dmesg --clear
dmesg | tee $LOG_FILE_BEFORE_LOAD-dmesg.log
$AMDXIO_PATH/AMDXIO -xgmilinkstatus 2>&1 | tee $LOG_FILE_BEFORE_LOAD.log
modprobe amdgpu
dmesg | tee $LOG_FILE_AFTER_LOAD-dmesg.log
$AMDXIO_PATH/AMDXIO -xgmilinkstatus 2>&1 | tee $LOG_FILE_AFTER_LOAD.log
