ATIFLASH_ABS_PATH=./debug-tool/atiflash
OUTPUT_FILE=/tmp/vbios-read.log
input="/tmp/vbios.log"

echo 		"============================" >  $OUTPUT_FILE
while IFS= read -r var
do
	echo 	"----------------------------" >> $OUTPUT_FILE
	echo 	"$var" >> $OUTPUT_FILE
	echo 	"----------------------------" >> $OUTPUT_FILE
	$ATIFLASH_ABS_PATH -biosfileinfo  $var >> $OUTPUT_FILE
done < "$input"
echo 		"============================" >>  $OUTPUT_FILE
$ATIFLASH_ABS_PATH -ai >> $OUTPUT_FILE
echo 		"============================" >>  $OUTPUT_FILE
cat $OUTPUT_FILE | grep "Bios P/N"
