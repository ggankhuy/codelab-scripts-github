LOG_FILE_CSSELECT=csselect.log
DB32_FILE=db32.mac
DB32_LOG=db32.log
DB32_LOG2=db32-2.log
DB32_SUMMARY=db32-summary.log

function countbits() {
    bitcount=0
    p1=$1
    for i in {0..32} ; do
        bitVal=$((p1 >> $i & 1))
        bitcount=$((bitcount+$bitVal))
    done
    echo bitcount for p1=$p1 : $bitcount
    return $bitcount
}


./db32 cmd csselect 2>&1 | tee $LOG_FILE_CSSELECT &

PID_LAST=$!
echo "PID: $PID_LAST"
sleep 3
kill $PID_LAST
sed -i 's/\*//g' $LOG_FILE_CSSELECT
echo "Reading back Display controllers only..."

while IFS= read -r line; do
    if [[ ! -z `echo $line | grep "Other Display Controller"` ]] ; then
        i=`echo $line | tr -s ' ' | cut -d ' ' -f1`
        echo "i: $i"
    	echo -ne "" > $DB32_FILE
	    echo csselect $i >> $DB32_FILE 
	    for j in 40000000 40010000 40020000 40030000 40040000 40050000 40060000 40070000 ; do
		    echo regw32 30800 $j >> $DB32_FILE
		    echo regr32 89bc >> $DB32_FILE
		    echo regr32 89c0 >> $DB32_FILE
	    done
	    echo "===========================" | tee -a $DB32_LOG
	    echo "device: $i" | tee -a $DB32_LOG
	    echo "device: $i" | tee -a $DB32_SUMMARY
	    echo "===========================" | tee -a $DB32_LOG
	    cat $DB32_FILE | grep csselect >> -a $DB32_LOG2
	    ./db32 exe $DB32_FILE 2>&1 | tee -a $DB32_LOG
        DB32_CURR_GPU_LOG=db32.$i.log
	    ./db32 exe $DB32_FILE 2>&1 | tee -a $DB32_LOG
        ./db32 exe $DB32_FILE 2>&1 | tee $DB32_CURR_GPU_LOG
        TOTAL_CG_DIS=0
        TOTAL_USER_DIS=0

        while IFS= read -r line; do
            echo "line: $line"
            if [[ ! -z `echo $line | grep 89bc` ]] ; then
                val=`echo $line | tr -s ' ' | cut -d ' ' -f6`
                countbits $val
                valBits=$?
                TOTAL_CG_DIS=$((TOTAL_CG_DIS + 0x$valBits))
            fi
            if [[ ! -z `echo $line | grep 89c0` ]] ; then
                val=`echo $line | tr -s ' ' | cut -d ' ' -f6`
                countbits $val
                valBits=$?
                TOTAL_USER_DIS=$((TOTAL_USER_DIS + 0x$valBits))
            fi
            echo "val: $val"
        done < $DB32_CURR_GPU_LOG
        echo "TOTAL CG   DISABLED: $TOTAL_CG_DIS" | tee -a  $DB32_SUMMARY
        echo "TOTAL USER DISABLED: $TOTAL_USER_DIS" | tee -a $DB32_SUMMARY
    fi
    
done < $LOG_FILE_CSSELECT
exit 0

