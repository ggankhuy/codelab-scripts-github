LOG_FILE_CSSELECT=csselect.log
FUSE_FILE=fuse.mac
FUSE_LOG=fuse.log
FUSE_LOG2=fuse2.log
FUSE_SUMMARY=fuse-summary.log
DEBUG=0
function countbits() {
    bitcount=0
    p1=$1
    for i in {0..32} ; do
        bitVal=$((((0x$p1)) >> $i))
        bitVal=$((bitVal &1 ))
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
echo -ne "" > $FUSE_SUMMARY
while IFS= read -r line; do
    if [[ ! -z `echo $line | grep "Other Display Controller"` ]] ; then
        i=`echo $line | tr -s ' ' | cut -d ' ' -f1`
        echo "i: $i"
    	echo -ne "" > $FUSE_FILE
	    echo csselect $i >> $FUSE_FILE 
	    for j in 40000000 40010000 40020000 40030000 40040000 40050000 40060000 40070000 ; do
		    echo regw32 30800 $j >> $FUSE_FILE
		    echo regr32 89bc >> $FUSE_FILE
		    echo regr32 89c0 >> $FUSE_FILE
	    done
	    echo "===========================" | tee -a $FUSE_LOG
	    echo "device: $i" | tee -a $FUSE_LOG
	    echo "device: $i" | tee -a $FUSE_SUMMARY
	    echo "===========================" | tee -a $FUSE_LOG
	    cat $FUSE_FILE | grep csselect >> -a $FUSE_LOG2
	    ./db32 exe $FUSE_FILE 2>&1 | tee -a $FUSE_LOG
        FUSE_CURR_GPU_LOG=fuse.$i.log
	    ./db32 exe $FUSE_FILE 2>&1 | tee -a $FUSE_LOG

        ./db32 exe $FUSE_FILE 2>&1 | tee $FUSE_CURR_GPU_LOG
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
        done < $FUSE_CURR_GPU_LOG
        echo "TOTAL CG   DISABLED: $TOTAL_CG_DIS" | tee -a  $FUSE_SUMMARY
        echo "TOTAL USER DISABLED: $TOTAL_USER_DIS" | tee -a $FUSE_SUMMARY
    fi
    
done < $LOG_FILE_CSSELECT

rm -rf fuse.*.log
rm -rf $LOG_FILE_CSSELECT
rm -rf $FUSE_LOG
clear

echo "---------------------------------"
echo "Fuse information:"
echo "---------------------------------"
cat $FUSE_SUMMARY
