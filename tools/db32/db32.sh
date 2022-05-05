LOG_FILE_CSSELECT=csselect.log


./db32 cmd csselect 2>&1 | tee $LOG_FILE_CSSELECT &

PID_LAST=$!
echo "PID: $PID_LAST"
sleep 3
kill $PID_LAST
sed -i 's/\*//g' $LOG_FILE_CSSELECT

echo "Reading back Display controllers only..."

while IFS= read -r line; do
    if [[ ! -z `echo $line | grep "Other Display Controller"` ]] ; then
        echo line: $line
        deviceNo=`echo $line | tr -s ' ' | cut -d ' ' -f1`
        echo "DeviceNo: $deviceNo"
    fi
done < $LOG_FILE_CSSELECT
exit 0

DB32_FILE=db32.mac
DB32_LOG=db32.log
DB32_LOG2=db32-2.log
for i in 52 55 95 98 174 180 217 223 ; do
	echo -ne "" > $DB32_FILE
	echo csselect $i >> $DB32_FILE 
	for j in 40000000 40010000 40020000 40030000 40040000 40050000 40060000 40070000 ; do
		echo regw32 30800 $j >> $DB32_FILE
		echo regr32 89bc >> $DB32_FILE
		echo regr32 89c0 >> $DB32_FILE
	done
	echo "===========================" | tee -a $DB32_LOG
	echo "device: $i" | tee -a $DB32_LOG
	echo "===========================" | tee -a $DB32_LOG
	cat $DB32_FILE | grep csselect >> -a $DB32_LOG2
	./db32 exe $DB32_FILE 2>&1 | tee -a $DB32_LOG
done
