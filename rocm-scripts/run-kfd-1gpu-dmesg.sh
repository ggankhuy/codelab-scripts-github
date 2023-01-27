KFD_SCRIPT=/usr/local/bin/run_kfdtest.sh
KFD=/usr/local/bin/kfdtest
DATE=`date +%Y%m%d-%H-%M-%S`
LOG_FOLDER=log/kfd/log-kfd-$DATE
mkdir $LOG_FOLDER -p
#LOG_SUMMARY=$LOG_FOLDER/summary.log
#echo -ne "" > $LOG_SUMMARY
#t1=$SECONDS

for i in {2..3} ; do
    #t1a=$SECONDS
    echo "Running kfd for gpu $i..." | tee -a $LOG_SUMMARY
    input="kfdtest-list.txt"
    while IFS= read -r line
    do
        echo "------------------"
        echo line: $line
        line=`echo $line | xargs`
        dmesg --clear
        LOG_FOLDER_CURR=$LOG_FOLDER/gpu$i
        mkdir -p $LOG_FOLDER_CURR
        LOG_FILE=$LOG_FOLDER_CURR/$line.test.log
        LOG_FILE_DMESG=$LOG_FOLDER_CURR/$line.dmesg.log
        echo LOG_FILE: $LOG_FILE
        echo LOG_FILE_DMESG: $LOG_FILE_DMESG
        echo $LOG_FILE | grep '\.\.'
        if [[ $? -eq 0 ]] ; then
            testgroup=$line            
        else
            echo "test command:"
            echo $KFD --gtest_filter=$testgroup$line 2>&1 | tee $LOG_FILE
            $KFD --gtest_filter=$testgroup$line 2>&1 | tee $LOG_FILE
            dmesg | tee $LOG_FILE_DMESG
        fi
    done < "$input"
    #t2a=$SECONDS
    #perGpuTime=$((t2a-t1a))
    #echo "gpu $i run time: $perGpuTime" | tee -a $LOG_SUMMARY
done
#t2=$SECONDS
#totalTime=$((t2-t1))
#echo "Total run time for all gpus: $totalTime" | tee -a $LOG_SUMMARY
