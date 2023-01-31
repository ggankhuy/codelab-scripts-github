KFD_SCRIPT=/usr/local/bin/run_kfdtest.sh
KFD=/usr/local/bin/kfdtest
DATE=`date +%Y%m%d-%H-%M-%S`
LOG_FOLDER=log/kfd/log-kfd-$DATE
mkdir $LOG_FOLDER -p
DEBUG=0
TEST_MODE=0
#LOG_SUMMARY=$LOG_FOLDER/summary.log
#echo -ne "" > $LOG_SUMMARY
#t1=$SECONDS

for i in {2..3} ; do
    #t1a=$SECONDS
    echo "Running kfd for gpu $i..." | tee -a $LOG_SUMMARY
    input="kfdtest-list.txt"
    counter=0
    while IFS= read -r line
    do
        echo "------------------"
        echo line: $line
        line=`echo $line | xargs`
        dmesg --clear
        LOG_FOLDER_CURR=$LOG_FOLDER/gpu$i
        mkdir -p $LOG_FOLDER_CURR
        LOG_FILE=$LOG_FOLDER_CURR/$counter-$testgroup$line.test.log
        LOG_FILE_DMESG=$LOG_FOLDER_CURR/$counter-$testgroup$line.dmesg.log
        echo $LOG_FILE | grep '\.\.'
        if [[ $? -eq 0 ]] ; then
            echo "setting testgroup to : $line"
            testgroup=$line            
        else
            echo LOG_FILE: $LOG_FILE
            echo LOG_FILE_DMESG: $LOG_FILE_DMESG
            bypass_flag=0
            for j in CacheInvalidateOnRemoteWrite LargestSysBufferTest LargestSysBufferTest CheckZeroInitializationVram;  do

                if [[ $DEBUG == 1 ]] ; then
                    echo "Checking if bypass: j: $j, line: $line:"
                fi
                if [[ $j == $line ]] ; then
                    echo "Setting $line to bypass"
                    if [[ $TEST_MODE == 1 ]] ; then
                        sleep 1
                    fi
                    bypass_flag=1
                fi
                if [[ $bypass_flag == 1 ]] ; then
                    break
                fi
            done
            if [[ $bypass_flag == "1" ]] ; then
                echo "Bypassing test: $line..."
                sleep 1
            else
                echo "test command:"
                echo "$KFD --gtest_filter=$testgroup$line 2>&1 | tee $LOG_FILE"
                sleep 1
                if [[ $TEST_MODE == 0 ]] ; then
                    echo "Launching test..."
                    $KFD --gtest_filter=$testgroup$line 2>&1 | tee $LOG_FILE
                fi
                dmesg | tee $LOG_FILE_DMESG
            fi
        fi
        counter=$((counter+1))
    done < "$input"
    
    #t2a=$SECONDS
    #perGpuTime=$((t2a-t1a))
    #echo "gpu $i run time: $perGpuTime" | tee -a $LOG_SUMMARY
done
#t2=$SECONDS
#totalTime=$((t2-t1))
#echo "Total run time for all gpus: $totalTime" | tee -a $LOG_SUMMARY
